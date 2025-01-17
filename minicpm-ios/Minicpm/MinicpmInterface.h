#ifndef MinicpmInterface_h
#define MinicpmInterface_h

#import <Foundation/Foundation.h>

@interface MinicpmInterface : NSObject

- (BOOL)initModel:(NSString *)llmPath projPath:(NSString *)projPath;
- (NSString *)completeWithText:(NSString *)text imageData:(NSData *)imageData;

@end

#endif /* MinicpmInterface_h */
