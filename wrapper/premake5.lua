workspace "MLIR_Benchmark"
  configurations { "Debug", "Release" }

platforms {"Static", "DLL"}


function capture_cmd(cmd)
    local f = assert(io.popen(cmd, 'r'))
    local s = assert(f:read('*a'))
    f:close()
    return s
end


python_include_path = os.getenv('CONDA_PREFIX') .. "/include/python3.11"
python_lib_path = os.getenv('CONDA_PREFIX') .. "/lib"
numpy_include_path = python_lib_path .. "/python3.11/site-packages/numpy/_core/include"
libffi_lib_path = "/usr/lib/x86_64-linux-gnu"


-- For MacOS, meaningless now (Since perf is not supported)
-- libffi_lib_path = "/opt/homebrew/opt/libffi/lib"
print(libffi_lib_path)



filter {"platforms:Static"}
   kind "StaticLib"
filter {"platforms:DLL"}
   kind "SharedLib"





-- ============================================
-- Utility project to build external libraries
-- ============================================
project "ext_build"
    kind "Makefile"
    location "build/ext"
    targetdir "lib"

    -- For each external dependency
    local ext_libs = { "perf-cpp"}
    local ext_root = path.getabsolute("ext")
    local lib_dir = path.getabsolute("lib")

    local build_cmds = {}
    for _, libname in ipairs(ext_libs) do
        local src_dir = path.join(ext_root, libname)
        local build_dir = path.join("build", "ext", libname)

        table.insert(build_cmds, "echo Building external lib: " .. libname)
        table.insert(build_cmds, "cmake -S " .. src_dir .. " -B " .. build_dir .. " -DCMAKE_BUILD_TYPE=%{cfg.buildcfg}")
        table.insert(build_cmds, "cmake --build " .. build_dir .. " --config %{cfg.buildcfg}")
        table.insert(build_cmds, "mkdir -p " .. lib_dir)
        table.insert(build_cmds, "find " .. build_dir .. " -name '*.so' -exec cp {} " .. lib_dir .. " \\;")
    end

    buildcommands(build_cmds)
    rebuildcommands { "echo Rebuilding external lib: "  }
    cleancommands { "rm -rf build/ext" }

project "WrapperModule"
   kind "ConsoleApp"
   language "C++"
   targetdir "build/%{cfg.buildcfg}"
   dependson {"ext_build"}


   
   includedirs { "./include/", numpy_include_path, python_include_path }
   libdirs { "./lib", python_lib_path , libffi_lib_path }

   links { "dl", "ffi", "python3.11" }


    -- This was for MacOS, but since perf is not supported here, this is meaningless 
   -- linkoptions {  "-framework CoreFoundation", "-Wl,-rpath," .. python_lib_path }
   linkoptions { "-Wl,-rpath," .. python_lib_path }


   files { "include/**.h", "src/**.cpp" }

   filter "configurations:Debug"
      buildoptions { "--std=c++20", "-g" } 
      defines { "DEBUG" }
      symbols "On"

   filter "configurations:Release"
      buildoptions { "--std=c++20" } 
      defines { "NDEBUG" }
      optimize "On"
