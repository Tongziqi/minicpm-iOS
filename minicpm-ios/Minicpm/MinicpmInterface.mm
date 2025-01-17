#import "MinicpmInterface.h"
#import "clip.h"

@implementation MinicpmInterface {
    struct clip_ctx* ctx;
}

- (instancetype)init {
    self = [super init];
    if (self) {
        ctx = nullptr;
    }
    return self;
}

- (void)dealloc {
    if (ctx) {
        clip_free(ctx);
    }
}

- (BOOL)initModel:(NSString *)llmPath projPath:(NSString *)projPath {
    if (ctx) {
        clip_free(ctx);
        ctx = nullptr;
    }
    
    const char* llmPathStr = [llmPath UTF8String];
    const char* projPathStr = [projPath UTF8String];
    
    ctx = clip_model_load(llmPathStr, projPathStr);
    return ctx != nullptr;
}

- (NSString *)completeWithText:(NSString *)text imageData:(NSData *)imageData {
    if (!ctx) {
        return @"Error: Model not initialized";
    }
    
    // Process image data and run inference
    // TODO: Implement actual completion logic
    return @"";
}

@end
