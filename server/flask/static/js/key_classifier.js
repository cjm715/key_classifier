
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
    let mic_rec = document.getElementById('mic_rec');
    let submit = document.getElementById('submit');
    let audioSave = document.getElementById('audioPlayer');
    let statusDiv = document.getElementById('statusDiv')
    let mediaRecorder = new MediaRecorder(mediaStreamObj);
    let chunks = [];
    var recording = false;
    let blob = new Blob();
    
    uploadBlob = function(){
        var form_data = new FormData();
        form_data.append('file', blob);

        $.ajax({
            type: 'POST',
            url: 'https://audiokey.net/predict_file',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            success: function(data) {
                console.log('Success!');
                let key = data.key
                let prob = data.probabilities[data.key]
                statusDiv.innerHTML = "The key is ".concat(data.key).concat(' with probability ').concat(prob);
                api_data = data;
            },
        })
    }

    mic_rec.addEventListener('click', (ev)=>{
        if (recording){
            mediaRecorder.stop();
            recording = false
            mic_rec.src = 'static/images/mic.svg'
            statusDiv.innerHTML = "Finished recording. Play recording below. If you are happy with the recording, submit to determine key."
        } else {
            mediaRecorder.start();
            recording = true
            mic_rec.src = 'static/images/record.svg'
            statusDiv.innerHTML = "recording ... (Stop by clicking button again)"
        }

        console.log(mediaRecorder.state);
    })
    submit.addEventListener('click', (ev)=>{
        statusDiv.innerHTML = "processing ..."
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


// // P5JS sketch for visualizing output from API
// let key_classifier = function(p){

//     p.setup = function() {
//         let cnv = p.createCanvas(300, 300);
//     }

//     p.draw = function() {
//         p.background('#f3f3f3')
//         if (api_data){
//             p.textSize(32);
//             p.text(api_data.key, 10, 30);
//         }
//     }

// }

// let myp5_3 = new p5(key_classifier, 'p5sketch');