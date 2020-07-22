# A flask api for face verification

## Requirements

1. System: 
    
        GCC version > v6.0
        libXext, libSM, libXrender, libstdc++
        conda version v3
        python version v3.7
    
2. pip wheels:

        tensorflow v1.14.0
        keras v2.2.4
        scipy
        numpy
        mtcnn
        Pillow
        matplotlib
        flask
        flask-cors
        keras-vggface
        
To download pip wheels:
```shell script
python install -r requiements
```

## vgg data sets

putmodel files under `~/.keras/models/vggface` directory.

[model files](https://github.com/rcmalli/keras-vggface/releases/tag/v2.0)

## settings

flask port is set to 11351

## description

api can take multiple files, and the first one is set to be the target to verify.