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
    @Published var messageLog: String = ""
    @Published var isInitialized = false
    let minicpm = MinicpmSwiftInterface()
    
    private var llamaContext: LlamaContext?
    
    func setBaseModel(model: LlamaModel?) {
        selectedBaseModel = model
    }
    
    public static func previewState() -> AppState {
        let ret = AppState()
        ret.downloadedBaseModels = [
            LlamaModel(name: "Local LLM Model", status: "downloaded", filename: "Model-7.6B-Q4_K_S.gguf", url: ""),
            LlamaModel(name: "Local Image Projection Model", status: "downloaded", filename: "Model-7.6B-Q4_K_S_mmproj-model-f16.gguf", url: "")
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
            
            let llmPath = Bundle.main.path(forResource: "Model-7.6B-Q4_K_S", ofType: "gguf") ?? ""
            let projPath = Bundle.main.path(forResource: "Model-7.6B-Q4_K_S_mmproj-model-f16", ofType: "gguf") ?? ""
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
    
    func loadModelsFromDisk() {
        print("Starting to load models from bundle...")
        
        // Add local models from bundle
        if let llmPath = Bundle.main.path(forResource: "Model-7.6B-Q4_K_S", ofType: "gguf") {
            print("Found LLM model at: \(llmPath)")
            downloadedBaseModels.append(LlamaModel(name: "Local LLM Model", status: "downloaded", filename: llmPath, url: ""))
        } else {
            print("Error: Could not find LLM model in bundle")
        }
        
        if let projPath = Bundle.main.path(forResource: "Model-7.6B-Q4_K_S_mmproj-model-f16", ofType: "gguf") {
            print("Found projection model at: \(projPath)")
            downloadedBaseModels.append(LlamaModel(name: "Local Image Projection Model", status: "downloaded", filename: projPath, url: ""))
        } else {
            print("Error: Could not find projection model in bundle")
        }
        
        print("Found \(downloadedBaseModels.count) models")
        if downloadedBaseModels.count == 2 {
            state = .Started
            print("Successfully loaded both models")
            
            #if targetEnvironment(simulator)
            print("Running on simulator, forcing CPU backend")
            #endif
            
            // Initialize models
            if let llmModel = downloadedBaseModels.first(where: { $0.name == "Local LLM Model" }),
               let projModel = downloadedBaseModels.first(where: { $0.name == "Local Image Projection Model" }) {
                let success = minicpm.initModel(llmModel.filename, projPath: projModel.filename)
                if !success {
                    print("Failed to initialize models")
                    messageLog = "Failed to initialize models"
                    state = .Startup
                }
            }
        } else {
            print("Error: Could not find all required models in app bundle")
        }
    }
    
    func preInit() async {
        do {
            if !isInitialized {
                await loadModelsFromDisk()
            }
        } catch {
            messageLog = "Error during initialization: \(error.localizedDescription)"
        }
    }
   
    func complete(text: String, img: Data?) async {
        guard let processedImg = img else {
            messageLog += "\nError: No image data provided"
            return
        }
        
        do {
            let result = minicpm.complete(text: text, imageData: processedImg)
            messageLog += result
        } catch {
            messageLog += "\nError during completion: \(error.localizedDescription)"
        }
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

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
