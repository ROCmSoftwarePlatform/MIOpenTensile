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
    def cmd = """
        rm -rf build
        mkdir build
        cd build
        CXX=/opt/rocm/hip/bin/hipcc cmake .. 
        make -j\$(nproc) ${config_targets}
    """
    echo cmd
    sh cmd
}

pipeline {
    agent{
        docker {
            image 'miopentensile'
            args '-v /var/jenkins/:/var/jenkins'
        }
    }
    stages{
        stage("Test"){
            agent{ label rocmnode("vega20") }
            steps{ buildJob("check") }
        }
        stage("Packaging") {
            agent{ label rocmnode("vega20") }
            steps{ buildJob("package")}
        }
    }
}