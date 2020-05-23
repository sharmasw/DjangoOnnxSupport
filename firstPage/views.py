from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
# Create your views here.
import onnx,onnxruntime,pathlib,json
from keras.preprocessing import image
import numpy as np

with open('imagenet_classes.json') as f:
    imageClassList=json.loads(f.read())


listOfmodels={}

def index(request):
    return render(request,'index.html')

def uploadModel(request):
    fileObj=request.FILES['filePath']
    fs=FileSystemStorage()
    filePathName=fs.save('models/'+fileObj.name,fileObj)
    filePathName=fs.url(filePathName)
    loadModel('.'+filePathName)
    context={'message':'Model Loaded Successful','listOfModels':list(listOfmodels.keys())}
    return render(request,'index.html',context)

def scoreImagePage(request):
    context={'message':'Model Loaded Successful','listOfModels':list(listOfmodels.keys())}
    return render(request,'scorepage.html',context)

def predictImage(request):
    fileObj=request.FILES['filePath']
    fs=FileSystemStorage()
    filePathName=fs.save('images/'+fileObj.name,fileObj)
    filePathName=fs.url(filePathName)
    modelName=request.POST.get('modelName')
    scorePrediction=predictImageData(modelName,'.'+filePathName)
    context={'scorePrediction':scorePrediction,'listOfModels':list(listOfmodels.keys())}
    return render(request,'scorepage.html',context)

def loadModel(filePath):
    fObj=pathlib.Path(filePath)
    onnx_model2 = onnx.load(filePath)
    content2 = onnx_model2.SerializeToString()
    sess2 = onnxruntime.InferenceSession(content2)
    listOfmodels[fObj.name]=sess2
    return 'Sucess'

def predictImageData(modelName,filePath):
    modelSess=listOfmodels[modelName]
    inpuInfo2=modelSess.get_inputs()[0]
    img_path=filePath# make sure the image is in img_path
    img = image.load_img(img_path, target_size=inpuInfo2.shape[1:3])
    testData = image.img_to_array(img)
    testData = np.expand_dims(testData, axis=0)
    testData.shape
    testData=testData/255
    inputData2=inpuInfo2.name,testData
    outputOFModel2=np.argmax(modelSess.run([modelSess.get_outputs()[0].name],dict([inputData2]))[0][0])
    score=imageClassList[str(outputOFModel2)]
    return score
