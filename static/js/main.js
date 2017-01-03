/**
 * Created by jasrub on 1/2/17.
 */


function updateTextInput(val, elId) {
    console.log("hi!");
    console.log(elId);
    console.log(document.getElementById(elId))
    document.getElementById(elId).value=val;
}

function changeSliderValue(val, sliderID){
    document.getElementById(sliderID).MaterialSlider.change(val);
}

$(function() {
    document.getElementById("filesPath").onchange = function () {
        document.getElementById("filesPathText").value = this.files[0].name;
    };

});

var veagan_parameters = {
    files: 'train/',
    learning_rate:0.0002,
    batch_size:64,
    n_epochs:25,
    n_examples:10,
    input_shape:[218, 178, 3],
    crop_shape:[64, 64, 3],
    crop_factor:0.8,
    n_filters:[100, 100, 100, 100],
    n_hidden:-1,
    n_code:128,
    convolutional: true,
    variational:true,
    filter_sizes:[3, 3, 3, 3],
    activation: 'tf.nn.elu',
    sample_step:100,
    ckpt_name: "vaegan.ckpt"
}

var dcgan_parameters = {
    "files": "train/",
    "epoch": 25,
    "learning_rate":0.0002,
    "beta1": 0.5,
    "batch_size": 64,
    "output_size": 64,
    "c_dim":3,
    "checkpoint_dir":"ckpt",
    "sample_dir":"samples",
    "sample_step":100
}

function train() {
        var inputs = dcgan_parameters;
        $.ajax({
                url: '/api/dcgan',
                method: 'POST',
                contentType: 'application/json',
                data: JSON.stringify(inputs),
                success: (data) => {console.log(data);}
    });
    }


