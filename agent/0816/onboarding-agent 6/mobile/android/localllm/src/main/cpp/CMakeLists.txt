cmake_minimum_required(VERSION 3.22.1)
project(llama_bridge)

add_library(llama_bridge SHARED
    llama_bridge_jni.cpp
)

# ==== llama.cpp linking (template) ====
# Provide your built static library and headers path via -DLLAMA_DIR=/path/to/llama.cpp
# Example build: cmake -S . -B build -DLLAMA_DIR=/absolute/path/llama.cpp -DHAS_LLAMA=ON
if(DEFINED LLAMA_DIR)
    message(STATUS "Using llama.cpp from: ${LLAMA_DIR}")
    add_library(llama_static STATIC IMPORTED)
    set_target_properties(llama_static PROPERTIES IMPORTED_LOCATION
        ${LLAMA_DIR}/build/android/${ANDROID_ABI}/libllama.a)

    target_include_directories(llama_bridge PRIVATE
        ${LLAMA_DIR}
        ${LLAMA_DIR}/include
        ${LLAMA_DIR}/ggml/include
    )
    target_link_libraries(llama_bridge PRIVATE llama_static log)
else()
    message(WARNING "LLAMA_DIR not set - building in demo mode without llama.cpp")
    target_compile_definitions(llama_bridge PRIVATE)
endif()

# Enable HAS_LLAMA when integrating real llama.cpp
# target_compile_definitions(llama_bridge PRIVATE HAS_LLAMA=1)
