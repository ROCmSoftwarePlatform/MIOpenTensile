def rocmnode(name) {
    def node_name = 'rocmtest'
    if(name == 'fiji') {
        node_name = 'rocmtest && fiji';
    } else if(name == 'vega') {
        node_name = 'rocmtest && vega';
    } else if(name == 'vega10') {
        node_name = 'rocmtest && vega10';
    } else if(name == 'vega20') {
        node_name = 'rocmtest && vega20';
    } else if(name == 'gfx908') {
        node_name = 'gfx908';
    } else {
        node_name = name
    }
    return node_name
}

def buildJob(config_targets="check"){
    retimage = docker.build("miopentensile")
    def cmd = ""
    if(config_targets == "package")
        cmd = """
            rm -rf build
            export HIPCC_LINK_FLAGS_APPEND='-O3 -parallel-jobs=4'
            export HIPCC_COMPILE_FLAGS_APPEND='-O3 -Wno-format-nonliteral -parallel-jobs=4'
            rbuild package -d deps --cxx /opt/rocm/hip/bin/hipcc
        """
    else
        cmd = """
            rm -rf build
            mkdir build
            cd build
            CXX=/opt/rocm/hip/bin/hipcc cmake .. 
            make -j check
        """

    withDockerContainer(image: "miopentensile"){
        echo cmd
        sh cmd
    }
    return retimage
}

pipeline{
    agent none
    stages{
        stage{
            parallel{
                stage("Test"){
                    agent{ label rocmnode("vega20") }
                    steps{ buildJob("check") }
                }
                stage("Packaging"){
                    agent{ label rocmnode("vega20") }
                    steps{ buildJob("package")}
                }
            }
        }
    }
}
