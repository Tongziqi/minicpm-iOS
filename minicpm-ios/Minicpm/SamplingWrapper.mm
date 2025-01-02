#import "SamplingWrapper.h"
#include "sampling.h"
#include "llava.hpp"
#include "common.hpp"
#include "clip.hpp"

@implementation SamplingWrapper {
    struct common_sampler *sampler;
    struct llama_context *llamaContext;
    int nPast;
}

- (instancetype)initWithLlamaCtx:(llama_context*)llamaCtx {
    self = [super init];
    if (self) {
        struct common_params_sampling params = {
            .temp = 0.7f,              // Match CLI temp
            .top_k = 100,             // Match CLI top_k
            .top_p = 0.8f,            // Match CLI top_p
            .min_p = 0.05f,
            .penalty_repeat = 1.05f,   // Match CLI repeat_penalty
            .penalty_last_n = 64,
            .n_prev = 64,
            .n_probs = 0,
            .min_keep = 0,
            .xtc_probability = 0.0f,
            .xtc_threshold = 0.1f,
            .typ_p = 1.0f,
            .dynatemp_range = 0.0f,
            .dynatemp_exponent = 1.0f,
            .penalty_freq = 0.0f,
            .penalty_present = 0.0f,
            .mirostat = 0,
            .mirostat_tau = 5.0f,
            .mirostat_eta = 0.1f,
            .ignore_eos = false,
            .seed = 0,
        };
        sampler = common_sampler_init(llama_get_model(llamaCtx), params);
        llamaContext = llamaCtx;
        nPast = 0;
    }
    return self;
}

- (void)resetSamplingContext {
    common_sampler_reset(sampler);
}

- (void)freeSamplingContext {
    common_sampler_free(sampler);
}

- (void)dealloc {
    [self freeSamplingContext];
}

std::vector<llama_token> llama_tokenize(
    const struct llama_model * model,
    const std::string & text,
    bool add_special,
    bool parse_special) {
    // upper limit for the number of tokens
    int n_tokens = text.length() + 2 * add_special;
    std::vector<llama_token> result(n_tokens);
    n_tokens = llama_tokenize(model, text.data(), text.length(), result.data(), result.size(), add_special, parse_special);
    if (n_tokens < 0) {
        result.resize(-n_tokens);
        int check = llama_tokenize(model, text.data(), text.length(), result.data(), result.size(), add_special, parse_special);
        GGML_ASSERT(check == -n_tokens);
    } else {
        result.resize(n_tokens);
    }
    return result;
}


- (NSArray<NSNumber *> *)tokenizeText:(NSString *)text addSpecial:(BOOL)addSpecial parseSpecial:(BOOL)parseSpecial {
    std::string str = std::string([text UTF8String]);
    auto tokens = llama_tokenize(llama_get_model(llamaContext), str, addSpecial, parseSpecial);
    
    NSMutableArray<NSNumber *> *result = [NSMutableArray arrayWithCapacity:tokens.size()];
    for (const auto token : tokens) {
        [result addObject:@(token)];
    }
    return result;
}

- (NSString *)tokenToPieceWithToken:(NSNumber *)token {
    char buf[8192];
    int32_t length = llama_token_to_piece(llama_get_model(llamaContext), 
                                        [token intValue],
                                        buf,
                                        sizeof(buf),
                                        0,  // no leading space stripping
                                        true); // render special tokens
    if (length < 0) {
        return @"";
    }
    return [[NSString alloc] initWithBytes:buf length:length encoding:NSUTF8StringEncoding];
}

- (BOOL)evaluateTokens:(NSArray<NSNumber *> *)tokens batchSize:(NSInteger)batchSize {
    if (tokens.count == 0) {
        return NO;  // Early return for empty tokens
    }
    
    std::vector<llama_token> tokensVec;
    tokensVec.reserve(tokens.count);
    for (NSNumber *token in tokens) {
        tokensVec.push_back([token intValue]);
    }
    
    struct llama_batch batch = llama_batch_init(tokensVec.size(), 0, 1);
    
    // Initialize batch with proper error checking
    if (!batch.token || !batch.pos || !batch.n_seq_id || !batch.seq_id || !batch.logits) {
        llama_batch_free(batch);
        return NO;
    }
    
    for (size_t i = 0; i < tokensVec.size(); ++i) {
        batch.token[i] = tokensVec[i];
        batch.pos[i] = nPast + i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i] = (llama_seq_id*)malloc(sizeof(llama_seq_id));
        if (!batch.seq_id[i]) {
            // Clean up previously allocated memory
            for (size_t j = 0; j < i; j++) {
                free(batch.seq_id[j]);
            }
            llama_batch_free(batch);
            return NO;
        }
        *batch.seq_id[i] = 0;
        batch.logits[i] = (i == tokensVec.size() - 1) ? 1 : 0;  // Only enable logits for last token
    }
    batch.n_tokens = tokensVec.size();
    
    int ret = llama_decode(llamaContext, batch);
    
    // Clean up allocated memory
    for (size_t i = 0; i < tokensVec.size(); ++i) {
        free(batch.seq_id[i]);
    }
    llama_batch_free(batch);
    
    if (ret != 0) {
        return NO;
    }
    
    nPast += tokensVec.size();
    return YES;
}

- (BOOL)embedImage:(UInt8*)image withSize:(int)size clipContext:(struct clip_ctx*)clip_ctx {
    struct llava_image_embed* embed = llava_image_embed_make_with_bytes(clip_ctx, 4, image, size);
    if (!embed || embed->n_image_pos <= 0) {
        if (embed) {
            llava_image_embed_free(embed);
        }
        return NO;
    }
    
    int batch_size = 16;  // Default batch size
    if (clip_is_minicpmv(clip_ctx)) {
        batch_size = embed->n_image_pos;  // For MINICPMV, process all tokens at once
        NSLog(@"MINICPMV model detected, processing all %d tokens at once", batch_size);
    }
    
    // Get context size
    const int n_ctx = llama_n_ctx(llamaContext);
    NSLog(@"Context size: %d", n_ctx);
    
    // Process embeddings in batches
    bool success = YES;
    for (int i = 0; i < embed->n_image_pos && success; i += batch_size) {
        int current_batch_size = std::min(batch_size, embed->n_image_pos - i);
        
        // Initialize batch with zero
        struct llama_batch batch = {
            .n_tokens = current_batch_size,
            .token = nullptr,  // token must be nullptr when using embeddings
            .embd = nullptr,
            .pos = nullptr,
            .n_seq_id = nullptr,
            .seq_id = nullptr,
            .logits = nullptr,
        };
        
        // Allocate memory for batch data
        batch.pos = (llama_pos*)calloc(current_batch_size, sizeof(llama_pos));
        batch.n_seq_id = (int32_t*)calloc(current_batch_size, sizeof(int32_t));
        batch.seq_id = (llama_seq_id**)calloc(current_batch_size, sizeof(llama_seq_id*));
        batch.logits = (int8_t*)calloc(n_ctx, sizeof(int8_t));  // Allocate for full context size
        
        if (!batch.pos || !batch.n_seq_id || !batch.seq_id || !batch.logits) {
            NSLog(@"Failed to allocate memory for batch metadata");
            success = NO;
            if (batch.pos) free(batch.pos);
            if (batch.n_seq_id) free(batch.n_seq_id);
            if (batch.seq_id) free(batch.seq_id);
            if (batch.logits) free(batch.logits);
            break;
        }
        
        // Allocate memory for embeddings
        const int embedding_size = 4096;  // CLIP's default embedding size
        batch.embd = (float*)calloc(current_batch_size * embedding_size, sizeof(float));
        if (!batch.embd) {
            NSLog(@"Failed to allocate memory for embeddings");
            success = NO;
            free(batch.pos);
            free(batch.n_seq_id);
            free(batch.seq_id);
            free(batch.logits);
            break;
        }
        
        // Copy embeddings
        memcpy(batch.embd, 
               embed->embed + (i * embedding_size), 
               current_batch_size * embedding_size * sizeof(float));
        
        // Allocate a single sequence ID for all tokens
        llama_seq_id* shared_seq_id = (llama_seq_id*)calloc(1, sizeof(llama_seq_id));
        if (!shared_seq_id) {
            NSLog(@"Failed to allocate memory for shared sequence ID");
            success = NO;
            free(batch.embd);
            free(batch.pos);
            free(batch.n_seq_id);
            free(batch.seq_id);
            free(batch.logits);
            break;
        }
        *shared_seq_id = 0;
        
        // Initialize batch data
        for (int j = 0; j < current_batch_size; j++) {
            batch.pos[j] = nPast + i + j;
            batch.n_seq_id[j] = 1;
            batch.seq_id[j] = shared_seq_id;  // Use the same sequence ID for all tokens
        }
        
        // Only enable logits for positions we're actually using
        memset(batch.logits, 0, n_ctx * sizeof(int8_t));  // First clear all
        for (int j = 0; j < current_batch_size; j++) {
            if (batch.pos[j] < n_ctx) {
                batch.logits[batch.pos[j]] = 1;  // Enable logits only for valid positions
            }
        }
        
        // Log batch status before decode
        NSLog(@"Current batch status before decode:");
        NSLog(@"- Number of tokens: %d", batch.n_tokens);
        NSLog(@"- First position: %d", batch.pos[0]);
        NSLog(@"- Last position: %d", batch.pos[current_batch_size - 1]);
        NSLog(@"- Batch size: %d", current_batch_size);
        NSLog(@"- Context size: %d", n_ctx);
        
        // Decode this batch
        int decode_result = llama_decode(llamaContext, batch);
        if (decode_result != 0) {
            NSLog(@"Failed to decode batch with error code: %d", decode_result);
            success = NO;
        }
        
        // Free resources
        free(shared_seq_id);
        free(batch.embd);
        free(batch.pos);
        free(batch.n_seq_id);
        free(batch.seq_id);
        free(batch.logits);
        
        if (!success) break;
    }
    
    // Free the original embedding
    llava_image_embed_free(embed);
    
    if (success) {
        nPast += embed->n_image_pos;
    }
    
    return success;
}

- (NSString *)sampleAndEvaluate {
    llama_token token = common_sampler_sample(sampler, llamaContext, nPast);
    common_sampler_accept(sampler, token, true);
    
    if (token == llama_token_eos(llama_get_model(llamaContext))) {
        return @"<image>";
    }
    
    if (token == llama_token_eos(llama_get_model(llamaContext))) {
        return @"</s>";
    }
    
    char buf[8192];
    int32_t length = llama_token_to_piece(llama_get_model(llamaContext), 
                                        token,
                                        buf,
                                        sizeof(buf),
                                        0,  // no leading space stripping
                                        true); // render special tokens
    if (length < 0) {
        return @"";
    }
    return [[NSString alloc] initWithBytes:buf length:length encoding:NSUTF8StringEncoding];
}

- (BOOL)evaluateString:(NSString *)string batchSize:(NSInteger)batchSize addBos:(BOOL)addBos {
    NSArray<NSNumber *> *tokens = [self tokenizeText:string addSpecial:addBos parseSpecial:YES];
    return [self evaluateTokens:tokens batchSize:batchSize];
}

@end
