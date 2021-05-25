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
    return node_name + '&& !nogpu';
}

def buildJob(config_targets="check"){
    retimage = docker.build("miopentensile")
    def cmd = ""
    if(config_targets == "package")
        cmd = """
            rm -rf build deps
            export HIPCC_LINK_FLAGS_APPEND='-O3 -parallel-jobs=4'
            export HIPCC_COMPILE_FLAGS_APPEND='-O3 -Wno-format-nonliteral -parallel-jobs=4'
            rbuild package -d deps --cxx /opt/rocm/hip/bin/hipcc -DCMAKE_INSTALL_PREFIX="/opt/rocm"
        """
    else
        cmd = """
            rm -rf build
            mkdir build
            cd build
            CXX=/opt/rocm/hip/bin/hipcc cmake -DCMAKE_INSTALL_PREFIX="/opt/rocm" .. 
            make -j check
        """

    withDockerContainer(image: "miopentensile", args: '-v=/var/jenkins/:/var/jenkins'){
        echo cmd
        sh cmd
    }
    return retimage
}

pipeline{
    agent none
    stages{
        stage("Test"){
            parallel{
                stage("Logic vega20"){
                    agent{ label rocmnode("vega20") }
                    steps{ buildJob("check") }
                }
                stage("Packaging vega20"){
                    agent{ label rocmnode("vega20") }
                    steps{ buildJob("package")}
                }
                stage("Logic gfx908"){
                    agent{ label rocmnode("gfx908") }
                    steps{ buildJob("check") }
                }
                stage("Packaging gfx908"){
                    agent{ label rocmnode("gfx908") }
                    steps{ buildJob("package")}
                }
            }
        }
    }
}
