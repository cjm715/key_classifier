
// GLOBAL VARIABLE THAT STORE API DATA
var api_data;


// AUDIO RECORDING AND QUERYING API
let constraintObj = { 
    audio: true, 
    video: false
}; 

if (navigator.mediaDevices === undefined) {
    navigator.mediaDevices = {};
    navigator.mediaDevices.getUserMedia = function(constraintObj) {
        let getUserMedia = navigator.webkitGetUserMedia || navigator.mozGetUserMedia;
        if (!getUserMedia) {
            return Promise.reject(new Error('getUserMedia is not implemented in this browser'));
        }
        return new Promise(function(resolve, reject) {
            getUserMedia.call(navigator, constraintObj, resolve, reject);
        });
    }
}else{
    navigator.mediaDevices.enumerateDevices()
    .then(devices => {
        devices.forEach(device=>{
            console.log(device.kind.toUpperCase(), device.label);
            //, device.deviceId
        })
    })
    .catch(err=>{
        console.log(err.name, err.message);
    })
}

navigator.mediaDevices.getUserMedia(constraintObj)
.then(function(mediaStreamObj) {
    //add listeners for saving video/audio
    let start = document.getElementById('btnStart');
    let stop = document.getElementById('btnStop');
    let audioSave = document.getElementById('audioPlayer');
    let submit = document.getElementById('btnSubmit');
    let keyDiv = document.getElementById('key')
    let mediaRecorder = new MediaRecorder(mediaStreamObj);
    let chunks = [];
    let blob = new Blob();
    
    uploadBlob = function(){
        var form_data = new FormData();
        form_data.append('file', blob);

        $.ajax({
            type: 'POST',
            url: 'https://18.144.173.139/predict_file',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            success: function(data) {
                console.log('Success!');
                let key = data.key
                let prob = data.probabilities[data.key]
                keyDiv.innerHTML = "The key is ".concat(data.key).concat(' with probability ').concat(prob);
                api_data = data;
            },
        })
    }

    start.addEventListener('click', (ev)=>{
        mediaRecorder.start();
        console.log(mediaRecorder.state);
    })
    stop.addEventListener('click', (ev)=>{
        mediaRecorder.stop();
        console.log(mediaRecorder.state);
    });
    submit.addEventListener('click', (ev)=>{
        uploadBlob()
    });

    mediaRecorder.ondataavailable = function(ev) {
        chunks.push(ev.data);
    }
    mediaRecorder.onstop = (ev)=>{
        blob = new Blob(chunks, {'type' : 'audio/wav' });
        chunks = [];
        let audioURL = window.URL.createObjectURL(blob);


        audioSave.src = audioURL;
    }
})
.catch(function(err) { 
    console.log(err.name, err.message); 
});


// P5JS sketch for visualizing output from API
let key_classifier = function(p){

    p.setup = function() {
        let cnv = p.createCanvas(300, 300);
    }

    p.draw = function() {
        p.background(255)
        if (api_data){
            p.textSize(32);
            p.text(api_data.key, 10, 30);
        }
    }

}

let myp5_3 = new p5(key_classifier, 'keyClassifierDiv');