// swift-tools-version: 5.9
import PackageDescription

let package = Package(
  name: "LocalLLM",
  defaultLocalization: "en",
  platforms: [
    .iOS(.v15)
  ],
  products: [
    .library(name: "LocalLLM", targets: ["LocalLLM"])
  ],
  targets: [
    // C bridge target that exposes llama_bridge.h to Swift
    .target(
      name: "CLlamaBridge",
      path: "bridge",
      publicHeadersPath: "include",
      sources: ["llama_bridge.c"],
      cSettings: [
        // Enable this define once you link against llama.cpp for iOS
        // .define("HAS_LLAMA"),
        .headerSearchPath("include")
      ],
      linkerSettings: [
        .linkedFramework("Metal"),
        .linkedFramework("Accelerate")
      ]
    ),
    // Swift wrapper that conforms to LocalLLM and uses CLlamaBridge
    .target(
      name: "LocalLLM",
      dependencies: ["CLlamaBridge"],
      path: ".",
      exclude: ["bridge", "Tests"],
      sources: ["LlamaLocalLLM.swift"]
    ),
    .testTarget(
      name: "LocalLLMTests",
      dependencies: ["LocalLLM"],
      path: "Tests",
      resources: []
    )
  ]
)
