import React, { Component } from "react";
import {
    Button,
    Slider,
    Grid,
    Cell,
    Layout,
    Header,
    Content,
    Textfield
} from "react-mdl";
import $ from 'jquery';

class ParameterRow extends React.Component {
    constructor(props) {
        super(props);
        this.handleChange = this.handleChange.bind(this);
    }

    handleChange(value) {
        this.props.onChange(value, this.props.idx);
    }

    render() {
        const value = this.props.parameter.value;
        const name = this.props.parameter.name;
        return (
            <tr>
                <td>{name}</td>
                <td>
                    <ParameterSlider
                        parameter={this.props.parameter}
                        value={+value}
                        onChange={this.handleChange} />
                </td>
                <td>
                    <ParameterText
                        value={""+value}
                        onChange={this.handleChange} />
                </td>
            </tr>
        );
    }
}

class ParameterText extends React.Component {
    constructor(props) {
        super(props);
        this.handleChange = this.handleChange.bind(this);
    }

    handleChange(e) {
        this.props.onChange(e.target.value);
    }
    render() {
        const value = this.props.value;
        return (
        <Textfield
            onChange={this.handleChange}
            pattern="-?[0-9]*(\.[0-9]+)?"
            error="Input is not a number!"
            style={{width: '50px'}}
            label={''}
            value={this.props.value}
        />
        );
    }

}

class ParameterSlider extends React.Component{
    constructor(props) {
        super(props);
        this.handleChange = this.handleChange.bind(this);
    }

    handleChange(e) {
        this.props.onChange(e.target.value);
    }
    render() {
        const min = this.props.parameter.min;
        const max = this.props.parameter.max;
        const step = this.props.parameter.step;
        const value = this.props.value;
        return (
            <Slider min={min} max={max} step={step} value={value} onChange={this.handleChange}/>
        );
    }
}

class ParametersList extends React.Component {
    constructor(props) {
        super(props);
        this.handleChange = this.handleChange.bind(this);
    }

    handleChange(value, idx) {
        this.props.onChange(value, idx);
        console.log(this.props.parameters)
    }
    render() {
        var rows = [];
        var lastCategory = null;
        let changeFunction = this.handleChange;
        this.props.parameters.forEach(function(parameter, index) {
            if (parameter.type===1) {
                rows.push(<ParameterRow parameter={parameter} key={parameter.name} idx={index} onChange={changeFunction}/>);
            }
        });
        return (
            <table style={{width: '100%'}}>
                <tbody>{rows}</tbody>
            </table>
        );
    }
}


class TrainingForm extends React.Component{
    constructor(props) {
        super(props);
        let training = false;
        this.state = {
            data: this.props.parameters,
            isTraining: training
        };

        this.handleSingleParameterChange = this.handleSingleParameterChange.bind(this);
        this.handleTrain = this.handleTrain.bind(this);
        this.handleSave = this.handleSave.bind(this);
    }

    handleSingleParameterChange(value, parameterIdx) {
        let newParameters = this.state.data;
        newParameters[parameterIdx].value = value;
        this.setState({data:newParameters});
        console.log("changed: "+newParameters[parameterIdx].name +"to"+value);
    }

    handleTrain(event) {
        train(this.state.data);
    }

    handleSave(event) {
        $.ajax({
            url: '/api/save_parameters',
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify(this.state.data),
            success: (data) => {
                console.log(data)
            }
        });
    }

    render() {
        return(
            <div>
                    <Grid>
                        <Cell col={12} >
                            <h4> Model Parameters </h4>
                            <ParametersList parameters={this.props.parameters} onChange={this.handleSingleParameterChange} />
                        </Cell>
                    </Grid>
                    <Grid>
                        <Cell col={6} >
                            <TrainButton handleTrain={this.handleTrain} disabled={this.props.isTraining}/>
                        </Cell>
                        <Cell col={6} >
                            <Button raised accent ripple onClick={this.handleSave}>Save Parameters</Button>
                        </Cell>
                    </Grid>
            </div>
        )
    }
}

class TrainButton extends React.Component {
    constructor(props) {
        super(props);
        this.handleTrain = this.handleTrain.bind(this);
    }

    handleTrain(event) {
        this.props.handleTrain();
    }
    render() {
        const isDisabled = this.props.disabled;

        let button = null;
        if (isDisabled) {
            button = <Button raised accent ripple disabled onClick={this.handleTrain}>Training...</Button>;
        } else {
            button = <Button raised accent ripple onClick={this.handleTrain}>Train</Button>;
        }

        return (
            <div>
                {button}
            </div>
        );
    }
}

var IframeComponent=React.createClass({
    render:function()
    {
        const training = this.props.isTraining;
        var Iframe=this.props.iframe;
        let view = null;
        console.log(training)
        if (!training) {
            view = <div> Tensorboard will show up here</div>;
        } else {

            view =
                <div>
                    <div> realod the page to update Tensorboard</div>
                    <Iframe src={this.props.src} height={this.props.height} width={this.props.width}/>
                </div>;
        }

        return(
            <div>
            { view }
            </div>
        );
    }
});

class TrainWindow extends React.Component {

    constructor(props) {
        super(props);
        let training = false;
        let tensorborad = false;
        this.state = {
            isTraining: training,
            tensorborad: tensorborad
        };
        this.setTrainingStatus = this.setTrainingStatus.bind(this);
    }

    setTrainingStatus() {
        var a=this;
        $.getJSON('/api/is_training', function( data ) {
            let isTraining = data['training'];
            let boardStatus = data['tensorboard'];
            a.setState({isTraining: isTraining, tensorborad:boardStatus})
        });
    }
    componentDidMount() {
        this.timerID = setInterval(
            () => this.setTrainingStatus(),
            200
        );
    }

    componentWillUnmount() {
        clearInterval(this.timerID);
    }

    render() {
        return(
        <Grid>
        <Cell col={4} shadow={4}>
            <TrainingForm parameters={dcgan_parameters} isTraining={this.state.isTraining}/>
        </Cell>
        <Cell col={8}>
            <IframeComponent iframe='iframe' src="http://127.0.0.1:6006" height="100%" width="100%" isTraining={this.state.tensorborad}/>
            </Cell>
        </Grid>
        )
    }
}

function train(inputParameters){
    $.ajax({
        url: '/api/dcgan',
        method: 'POST',
        contentType: 'application/json',
        data: JSON.stringify(inputParameters),
        success: (data) => {
           console.log(data)
        }
    });
}

var VAEGAN = [
    {name: 'files', value:'train/', type:5},
    {name: 'learning_rate', value:0.0002, min:0, max:0.1, step:0.00000001, type:1},
    {name: 'batch_size' ,value:64, min:3, max:300, step:1, type:1},
    {name: 'n_epochs', value:25, min:1, max:300, step:1, type:1},
    {name: 'n_examples', value:10, min:0, max:300, step:1, type:1},
    {name: 'input_shape', value:[218, 178, 3], type:2},
    {name: 'crop_shape', value:[64, 64, 3], type:2},
    {name: 'crop_factor', value:0.8, min:0.2, max:1, step:0.01, type:1},
    {name: 'n_filters', value:[100, 100, 100, 100], type:2},
    {name: 'n_hidden', value:-1, type:1, min:-1, max:30, step:1},
    {name: 'n_code', value:128, min:1, max:400, step:1, type:1},
    {name: 'convolutional', value: true, type:3},
    {name: 'variational', value:true, type:3},
    {name: 'filter_sizes', value:[3, 3, 3, 3], type:2},
    {name: 'activation', value: 'tf.nn.elu', type:4},
    {name: 'sample_step', value:100, min:1, max:1000, step:1, type:1},
    {name: 'ckpt_name', value: "vaegan.ckpt", type:5}
];
// type: 1- number, 2-array, 3-bool, 4-function, 5-path

var dcgan_parameters = [
    {name: "dataset", value: "train", type:5},
    {name: "epoch", value: 25, type:1, min:1, max:300, step:1},
    {name: "learning_rate", value:0.0002, min:0, max:0.1, step:0.00000001, type:1},
    {name:"beta1", value:0.5, min:0, max:1, step:0.0001, type:1},
    {name:"batch_size", value:64, min:3, max:300, step:1, type:1},
    {name:"output_size", value:64, min:10, max:500, step:1, type:1},
    {name:"c_dim", value:3, min:1, max:10, step:1, type:1},
    {name:"checkpoint_dir", value:"ckpt", type:5},
    {name:"sample_dir", value:"samples", type:5},
    {name:'image_size', value:64, min:10, max:500, step:1, type:1},
    {name:'is_train', value:'True', type:3},
    {name:'visualize', value:'False', type:3}
    //"sample_step":100
];


class App extends Component {
    render() {
        return (
        <div>
            <Layout fixedHeader>
                <Header title="GooeyBrain" className="mdl-color--teal-800">
                </Header>
                <Content>
                    <TrainWindow/>
                </Content>
            </Layout>
        </div>
        );
    }
}

export default App;