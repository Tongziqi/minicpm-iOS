import Foundation

enum LlamaError: Error {
    case couldNotInitializeContext
    case completionError(String)
}

func llama_batch_clear(_ batch: inout llama_batch) {
    batch.n_tokens = 0
}

func llama_batch_add(_ batch: inout llama_batch, _ id: llama_token, _ pos: llama_pos, _ seq_ids: [llama_seq_id], _ logits: Bool) {
    let currentIndex = Int(batch.n_tokens)
    
    // 确保我们不会越界访问数组
    if currentIndex >= 2048 { // 使用初始化时设置的固定大小
        return
    }
    
    batch.token   [currentIndex] = id
    batch.pos     [currentIndex] = pos
    batch.n_seq_id[currentIndex] = Int32(seq_ids.count)
    for i in 0..<seq_ids.count {
        batch.seq_id[currentIndex]![Int(i)] = seq_ids[i]
    }
    // Always enable logits for MINICPMV models
    batch.logits  [currentIndex] = 1
    
    batch.n_tokens += 1
}

struct LlamaParams {
    var nPredict: Int = 256
    var nBatch: Int = 2048
}

actor LlamaContext {
    private var model: OpaquePointer
    private var ctx: OpaquePointer
    private var batch: llama_batch
    private var tokens_list: [llama_token]
    private var clip_ctx: OpaquePointer?
    private var systemPrompt: String
    private var userPromptPostfix: String
    private var sampling_ctx: SamplingWrapper
    private var n_past: Int32 = 0
    private var needsSystemInit = true
    
    /// This variable is used to store temporarily invalid cchars
    private var temporary_invalid_cchars: [CChar]
    
    var n_len: Int32 = 2048
    var n_cur: Int32 = 0
    
    var n_decode: Int32 = 0
    
    init(model: OpaquePointer, ctx: OpaquePointer, clip_ctx: OpaquePointer?, systemPrompt: String, userPromptPostfix: String) {
        self.model = model
        self.ctx = ctx
        self.clip_ctx = clip_ctx
        self.tokens_list = []
        // Increase batch size to handle MINICPMV models
        let maxBatchSize = Int32(2048)
        self.batch = llama_batch_init(maxBatchSize, 0, 1)
        self.temporary_invalid_cchars = []
        self.systemPrompt = systemPrompt
        self.userPromptPostfix = userPromptPostfix
        self.sampling_ctx = SamplingWrapper(llamaCtx: ctx)
    }
    
    deinit {
        if clip_ctx != nil {
            clip_free(clip_ctx)
        }
        llama_batch_free(batch)
        llama_free(ctx)
        llama_free_model(model)
        llama_backend_free()
    }
    
    static func create_context(path: String, clipPath: String?, systemPrompt: String, userPromptPostfix: String) throws -> LlamaContext {
        llama_backend_init()
        var model_params = llama_model_default_params()
        
#if targetEnvironment(simulator)
        model_params.n_gpu_layers = 0
        print("Running on simulator, force use n_gpu_layers = 0")
#endif
        guard let model = llama_load_model_from_file(path, model_params) else {
            print("Could not load model at \(path)")
            throw LlamaError.couldNotInitializeContext
        }
        
        let clip_ctx = clipPath.flatMap { clipPath in
            clip_model_load(clipPath, 1)
        }
        
        let n_threads = max(1, min(4, ProcessInfo.processInfo.processorCount - 1))
        print("Using \(n_threads) threads")
        
        var ctx_params = llama_context_default_params()
        ctx_params.n_ctx = 2048  // Reduced context size for iOS
        ctx_params.n_threads = Int32(n_threads)
        ctx_params.n_threads_batch = Int32(n_threads)
        
        guard let ctx = llama_new_context_with_model(model, ctx_params) else {
            print("Could not load context!")
            throw LlamaError.couldNotInitializeContext
        }
        
        return LlamaContext(model: model, ctx: ctx, clip_ctx: clip_ctx, systemPrompt: systemPrompt, userPromptPostfix: userPromptPostfix)
    }
    
    func model_info() -> String {
        let result = UnsafeMutablePointer<Int8>.allocate(capacity: 256)
        result.initialize(repeating: Int8(0), count: 256)
        defer {
            result.deallocate()
        }
        
        let nChars = llama_model_desc(model, result, 256)
        let bufferPointer = UnsafeBufferPointer(start: result, count: Int(nChars))
        
        var SwiftString = ""
        for char in bufferPointer {
            SwiftString.append(Character(UnicodeScalar(UInt8(char))))
        }
        
        return SwiftString
    }
    
    func get_n_tokens() -> Int32 {
        return batch.n_tokens
    }
    
    func completion_init(text: String, imageBytes: [UInt8]?) async {
        print("attempting to complete \"\(text)\"")
        if needsSystemInit {
            await completion_system_init()
        }
        
        // Reset batch and KV cache before processing new input
        llama_batch_clear(&batch)
        llama_kv_cache_clear(ctx)
        
        if let imageBytes = imageBytes {
            var myBytes = imageBytes
            let size = Int32(myBytes.count)
            print("Processing image with size: \(size) bytes")
            
            let success = myBytes.withUnsafeMutableBytes { raw in
                sampling_ctx.embedImage(raw.baseAddress, withSize: size, clipContext: clip_ctx)
            }
            print("Image embedding success: \(success)")
            
            // Verify batch status after image embedding
            if batch.n_tokens > 0 {
                // Enable logits for all tokens
                for i in 0..<Int(batch.n_tokens) {
                    batch.logits[i] = 1
                }
                
                // Evaluate the batch in smaller chunks
                let chunkSize = 32
                var processedTokens = 0
                
                while processedTokens < batch.n_tokens {
                    let remainingTokens = Int(batch.n_tokens) - processedTokens
                    let currentChunkSize = min(chunkSize, remainingTokens)
                    
                    var tempBatch = llama_batch_init(Int32(currentChunkSize), 0, 1)
                    defer { llama_batch_free(tempBatch) }
                    
                    // Copy tokens to temp batch
                    for i in 0..<currentChunkSize {
                        let srcIdx = processedTokens + i
                        tempBatch.token[i] = batch.token[srcIdx]
                        tempBatch.pos[i] = batch.pos[srcIdx]
                        tempBatch.n_seq_id[i] = batch.n_seq_id[srcIdx]
                        tempBatch.logits[i] = 1
                        tempBatch.n_tokens += 1
                    }
                    
                    if llama_decode(ctx, tempBatch) != 0 {
                        print("Error: Failed to decode batch chunk")
                    }
                    
                    processedTokens += currentChunkSize
                }
            }
        }
        
        // Initialize sampling context with the new input
        sampling_ctx.evaluateString("\(text)\(userPromptPostfix)", batchSize: 512, addBos: false)
    }
    
    func completion_loop() async throws -> String {
        needsSystemInit = true
        
        // Ensure we have valid context and batch
        if ctx == nil {
            throw LlamaError.completionError("Context is nil")
        }
        
        // Sample and evaluate
        guard let ret = sampling_ctx.sampleAndEvaluate() else {
            throw LlamaError.completionError("Failed to sample and evaluate")
        }
        
        n_cur += 1
        return ret
    }
    
    func completion_system_init() async {
        llama_batch_clear(&batch)
        llama_kv_cache_clear(ctx)
        sampling_ctx.evaluateString(systemPrompt, batchSize: 2048, addBos: true)
        needsSystemInit = false
    }
    
    func clear() async {
        tokens_list.removeAll()
        temporary_invalid_cchars.removeAll()
        llama_kv_cache_clear(ctx)
        sampling_ctx.resetSamplingContext()
    }
}
