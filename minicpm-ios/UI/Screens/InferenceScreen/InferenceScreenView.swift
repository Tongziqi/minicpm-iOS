//
//  ModelSelectScreen.swift
//  llava-ios
//
//  Created by Prashanth Sadasivan on 2/12/24.
//

import SwiftUI
import Vision

struct InferenceScreenView: View {
    @StateObject var appstate: AppState
    @State private var cameraModel: CameraDataModel
    @State private var showingImagePicker = false
    @State private var showingActionSheet = false
    @State private var selectedImage: UIImage?
    @State private var sourceType: UIImagePickerController.SourceType = .photoLibrary
    @State private var isInferencing = false
    @State private var errorMessage: String?
    
    init(appstate: AppState, cameraModel: CameraDataModel) {
        self._appstate = StateObject(wrappedValue: appstate)
        self.cameraModel = cameraModel
    }
    
    var body: some View {
        VStack(spacing: 20) {
            // 标题
            Text("MiniCPM Demo")
                .font(.title)
                .fontWeight(.bold)
                .padding(.top)
            
            // 图片显示区域
            if let selectedImage = selectedImage {
                Image(uiImage: selectedImage)
                    .resizable()
                    .scaledToFit()
                    .frame(maxHeight: 300)
            } else if (self.cameraModel.currentImageData != nil) {
                CameraView(model: self.cameraModel)
                    .frame(maxHeight: 300)
            } else {
                Color.gray.opacity(0.2)
                    .frame(height: 300)
                    .overlay(Text("No image selected"))
            }
            
            // 按钮区域
            HStack(spacing: 20) {
                Button("Select Image") {
                    showingActionSheet = true
                }
                .disabled(isInferencing)
                
                if isInferencing {
                    ProgressView()
                        .progressViewStyle(CircularProgressViewStyle())
                        .scaleEffect(1.2)
                } else {
                    Button("Inference") {
                        sendText()
                    }
                    .disabled(selectedImage == nil && cameraModel.currentImageData == nil)
                }
            }
            .buttonStyle(.bordered)
            
            // 推理结果或错误显示区域
            if let error = errorMessage {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Error:")
                        .font(.headline)
                        .foregroundColor(.red)
                    Text(error)
                        .font(.system(size: 14))
                        .foregroundColor(.red)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding()
                .background(
                    RoundedRectangle(cornerRadius: 10)
                        .fill(Color(.systemBackground))
                        .shadow(radius: 2)
                )
            } else if !appstate.messageLog.isEmpty {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Inference Result:")
                        .font(.headline)
                    Text(appstate.messageLog)
                        .font(.system(size: 14))
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding()
                .background(
                    RoundedRectangle(cornerRadius: 10)
                        .fill(Color(.systemBackground))
                        .shadow(radius: 2)
                )
            }
            
            Spacer()
        }
        .padding()
        .sheet(isPresented: $showingImagePicker) {
            ImagePicker(image: $selectedImage, sourceType: sourceType)
        }
        .actionSheet(isPresented: $showingActionSheet) {
            ActionSheet(title: Text("Select Image Source"), buttons: [
                .default(Text("Camera")) {
                    sourceType = .camera
                    clearResults()
                    showingImagePicker = true
                },
                .default(Text("Photo Library")) {
                    sourceType = .photoLibrary
                    clearResults()
                    showingImagePicker = true
                },
                .cancel()
            ])
        }
        .onChange(of: selectedImage) { _ in
            clearResults()
        }
        .onAppear {
            Task {
               await appstate.preInit()
            }
        }
    }
    
    func clearResults() {
        errorMessage = nil
        appstate.messageLog = ""
    }
    
    func resizeImage(_ image: UIImage, targetSize: CGSize) async throws -> UIImage {
        guard let cgImage = image.cgImage else {
            throw NSError(domain: "ImageProcessing", code: -1, userInfo: [NSLocalizedDescriptionKey: "Failed to get CGImage"])
        }
        
        let requestHandler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        let request = VNGenerateImageFeaturePrintRequest()
        try requestHandler.perform([request])
        
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        
        let renderer = UIGraphicsImageRenderer(size: targetSize, format: format)
        let resizedImage = renderer.image { context in
            image.draw(in: CGRect(origin: .zero, size: targetSize))
        }
        
        return resizedImage
    }
    
    func sendText() {
        Task {
            isInferencing = true
            errorMessage = nil
            
            do {
                if let selectedImage = selectedImage {
                    // 直接压缩到 448x448
                    let resizedImage = try await resizeImage(selectedImage, targetSize: CGSize(width: 448, height: 448))
                    if let imageData = resizedImage.jpegData(compressionQuality: 0.8) {
                        await appstate.complete(text: "", img: imageData)
                    } else {
                        throw NSError(domain: "ImageProcessing", code: -2, userInfo: [NSLocalizedDescriptionKey: "Failed to convert image to data"])
                    }
                } else if let currentImageData = cameraModel.currentImageData {
                    // 处理相机图片
                    if let image = UIImage(data: currentImageData) {
                        let resizedImage = try await resizeImage(image, targetSize: CGSize(width: 448, height: 448))
                        if let imageData = resizedImage.jpegData(compressionQuality: 0.8) {
                            await appstate.complete(text: "", img: imageData)
                        } else {
                            throw NSError(domain: "ImageProcessing", code: -2, userInfo: [NSLocalizedDescriptionKey: "Failed to convert camera image to data"])
                        }
                    } else {
                        throw NSError(domain: "ImageProcessing", code: -3, userInfo: [NSLocalizedDescriptionKey: "Failed to process camera image"])
                    }
                } else {
                    errorMessage = "No image selected"
                }
                
                // 检查推理结果是否包含错误信息
                if appstate.messageLog.contains("Error during completion loop:") {
                    errorMessage = "Inference failed. Please try again with a different image."
                }
            } catch {
                errorMessage = "Error: \(error.localizedDescription)"
            }
            
            isInferencing = false
        }
    }
}

struct ImagePicker: UIViewControllerRepresentable {
    @Binding var image: UIImage?
    var sourceType: UIImagePickerController.SourceType
    
    func makeUIViewController(context: Context) -> UIImagePickerController {
        let picker = UIImagePickerController()
        picker.delegate = context.coordinator
        picker.sourceType = sourceType
        return picker
    }
    
    func updateUIViewController(_ uiViewController: UIImagePickerController, context: Context) {}
    
    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }
    
    class Coordinator: NSObject, UIImagePickerControllerDelegate, UINavigationControllerDelegate {
        let parent: ImagePicker
        
        init(_ parent: ImagePicker) {
            self.parent = parent
        }
        
        func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
            if let image = info[.originalImage] as? UIImage {
                parent.image = image
            }
            picker.dismiss(animated: true)
        }
        
        func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
            picker.dismiss(animated: true)
        }
    }
}
