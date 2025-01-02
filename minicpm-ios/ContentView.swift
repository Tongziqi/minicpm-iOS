//
//  ContentView.swift
//  llava-ios
//
//  Created by Prashanth Sadasivan on 2/2/24.
//

import SwiftUI
import Vision
import CoreImage

struct LlamaModel: Identifiable{
    var id = UUID()
    var name: String
    var status: String
    var filename: String
    var url: String
}

class AppState: ObservableObject {
    
    static var llavaModels = LlavaModelInfoList(models: [
        LlavaModelInfo(modelName: "Local LLM Model", url: "", projectionUrl: "")
    ])
    
    let NS_PER_S = 1_000_000_000.0
    enum StartupState{
        case Startup
        case Loading
        case Started
    }
    
    var selectedBaseModel: LlamaModel?
    @Published var downloadedBaseModels: [LlamaModel] = []
    @Published var state: StartupState = .Startup
    @Published var useTiny: Bool = true
    @Published var messageLog = ""
    
    private var llamaContext: LlamaContext?
    
    func setBaseModel(model: LlamaModel?) {
        selectedBaseModel = model
    }
    
    public static func previewState() -> AppState {
        let ret = AppState()
        ret.downloadedBaseModels = [
            LlamaModel(name: "Local LLM Model", status: "downloaded", filename: "ggml-model-Q2_K.gguf", url: ""),
            LlamaModel(name: "Local Image Projection Model", status: "downloaded", filename: "mmproj-model-f16.gguf", url: "")
        ]
        return ret
    }
    
    func appendMessage(result: String) {
        DispatchQueue.main.async {
            self.messageLog += "\(result)"
        }
    }
    
    var TINY_SYS_PROMPT: String = "<|system|>\n What is in the image.</s>\n\n"
    
    var TINY_USER_POSTFIX: String = "</s>\n<|assistant|>\n"
    
    var DEFAULT_SYS_PROMPT: String = "USER:"
    var DEFAULT_USER_POSTFIX: String = "\nASSISTANT:"
    
    func ensureContext() {
        if llamaContext == nil {
            print("loading MINICPMV model")
            
            let llmPath = Bundle.main.path(forResource: "ggml-model-Q2_K", ofType: "gguf") ?? ""
            let projPath = Bundle.main.path(forResource: "mmproj-model-f16", ofType: "gguf") ?? ""
            print("Model paths - LLM: \(llmPath), Proj: \(projPath)")
            
            if !llmPath.isEmpty && !projPath.isEmpty {
                print("GOT THE MODELS")
                do {
                    let systemPrompt = ""  // Remove system prompt to match CLI
                    let userPromptPostfix = ""  // Remove postfix to match CLI
                    
                    self.llamaContext = try LlamaContext.create_context(path: llmPath, clipPath: projPath, systemPrompt: systemPrompt, userPromptPostfix: userPromptPostfix)
                } catch {
                    messageLog += "Error loading models: \(error)\n"
                    return
                }
            } else {
                print("MISSING MODELS")
                messageLog += "Error: Could not find models in app bundle\n"
            }
        }
        guard let llamaContext else {
            return
        }
    }
    
    func preInit() async {
        ensureContext()
        guard let llamaContext else {
            return
        }
        await llamaContext.completion_system_init()
    }
   
    func complete(text: String, img: Data?) async {
        ensureContext()
        guard let llamaContext else {
            appendMessage(result: "\nError: LLaMA context not initialized\n")
            return
        }
        
        let processedImage: Data?
        if let img = img {
            do {
                processedImage = try await ImageProcessor.preprocessImageWithVision(img)
            } catch {
                appendMessage(result: "\nError processing image: \(error.localizedDescription)\n")
                return
            }
        } else {
            processedImage = nil
        }
        
        let bytes = processedImage.map { d in
            var byteArray = [UInt8](repeating: 0, count: d.count)
            d.copyBytes(to: &byteArray, count: d.count)
            return byteArray
        }
        
        let t_start = DispatchTime.now().uptimeNanoseconds
        do {
            await llamaContext.completion_init(text: text, imageBytes: bytes)
            let t_heat_end = DispatchTime.now().uptimeNanoseconds
            let t_heat = Double(t_heat_end - t_start) / NS_PER_S
            appendMessage(result: "\(text)")
            
            var shouldContinue = true
            while shouldContinue {
                do {
                    let n_cur = await llamaContext.n_cur
                    let n_len = await llamaContext.n_len
                    
                    if n_cur >= n_len {
                        break
                    }
                    
                    let result = try await llamaContext.completion_loop()
                    if result.isEmpty {
                        shouldContinue = false
                        continue
                    }
                    
                    appendMessage(result: "\(result)")
                    if result == "</s>" || result == "<|im_end|>" {
                        shouldContinue = false
                    }
                } catch {
                    appendMessage(result: "\nError during completion loop: \(error.localizedDescription)\n")
                    shouldContinue = false
                }
            }
            
            let t_end = DispatchTime.now().uptimeNanoseconds
            let t_generation = Double(t_end - t_heat_end) / NS_PER_S
            let n_cur = await llamaContext.n_cur
            let tokens_per_second = Double(n_cur) / t_generation
            
            await llamaContext.clear()
            await llamaContext.completion_system_init()
            appendMessage(result: """
                \n
                Done
                Heat up took \(t_heat)s
                Generated \(tokens_per_second) t/s\n
                """
                          )
        } catch {
            appendMessage(result: "\nError during completion: \(error.localizedDescription)\n")
            await llamaContext.clear()
            await llamaContext.completion_system_init()
        }
    }
    
    func clear() async {
        guard let llamaContext else {
            return
        }

        await llamaContext.clear()
        DispatchQueue.main.async {
            self.messageLog = ""
        }
    }

    public func loadModelsFromDisk() {
        print("Starting to load models from bundle...")
        
        // Add local models from bundle
        if let llmPath = Bundle.main.path(forResource: "ggml-model-Q2_K", ofType: "gguf") {
            print("Found LLM model at: \(llmPath)")
            downloadedBaseModels.append(LlamaModel(name: "Local LLM Model", status: "downloaded", filename: llmPath, url: ""))
        } else {
            print("Error: Could not find LLM model in bundle")
        }
        
        if let projPath = Bundle.main.path(forResource: "mmproj-model-f16", ofType: "gguf") {
            print("Found projection model at: \(projPath)")
            downloadedBaseModels.append(LlamaModel(name: "Local Image Projection Model", status: "downloaded", filename: projPath, url: ""))
        } else {
            print("Error: Could not find projection model in bundle")
        }
        
        print("Found \(downloadedBaseModels.count) models")
        if downloadedBaseModels.count == 2 {
            state = .Started
            print("Successfully loaded both models")
        } else {
            print("Error: Could not find all required models in app bundle")
        }
    }
    
    
    func getDocumentsDirectory() -> URL {
        let paths = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)
        return paths[0]
    }
}

extension UIImage {
    func resizedForClip() -> UIImage? {
        let targetSize = CGSize(width: 224, height: 224)  // CLIP 标准输入尺寸
        
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        
        let renderer = UIGraphicsImageRenderer(size: targetSize, format: format)
        let resizedImage = renderer.image { _ in
            self.draw(in: CGRect(origin: .zero, size: targetSize))
        }
        
        return resizedImage
    }
}

class ImageProcessor {
    static func processImageForClip(_ imageData: Data) -> Data? {
        guard let image = UIImage(data: imageData) else { return nil }
        guard let resizedImage = image.resizedForClip() else { return nil }
        return resizedImage.pngData()
    }
    
    static func preprocessImageWithVision(_ imageData: Data) async throws -> Data {
        guard let image = UIImage(data: imageData) else {
            throw NSError(domain: "ImageProcessing", code: -1, userInfo: [NSLocalizedDescriptionKey: "Failed to create image from data"])
        }
        
        guard let cgImage = image.cgImage else {
            throw NSError(domain: "ImageProcessing", code: -2, userInfo: [NSLocalizedDescriptionKey: "Failed to get CGImage"])
        }
        
        let requestHandler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        
        let request = VNGenerateImageFeaturePrintRequest()
        try requestHandler.perform([request])
        
        // 调整图像尺寸为 CLIP 模型需要的大小
        guard let resizedImage = image.resizedForClip(),
              let processedData = resizedImage.pngData() else {
            throw NSError(domain: "ImageProcessing", code: -3, userInfo: [NSLocalizedDescriptionKey: "Failed to resize image"])
        }
        
        return processedData
    }
}

struct ContentView: View {
    @StateObject var appstate = AppState()
    @State private var cameraModel = CameraDataModel()
    var body: some View {
        TabView {
            VStack {
                if appstate.state == .Startup {
                    Text("Loading....")
                } else if appstate.state == .Started {
                    InferenceScreenView(appstate: appstate, cameraModel: cameraModel)
                }
            }
        }
        .padding()
        .onAppear {
            appstate.state = .Loading
            appstate.loadModelsFromDisk()
        }
    }
    
}

//#Preview {
//    ContentView()
//}
