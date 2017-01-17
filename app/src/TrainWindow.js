import React, { Component } from "react";
import {
    Button,
    Slider,
    Grid,
    Cell,
    Layout,
    Header,
    Content,
    Textfield,
    Switch
} from "react-mdl";
import $ from 'jquery';
import ParametersList from './Parameters';

class TrainingForm extends React.Component{
    constructor(props) {
        super(props);
        let training = false;
        this.state = {
            data: [],
            isTraining: training
        };

        this.handleSingleParameterChange = this.handleSingleParameterChange.bind(this);
        this.handleTrain = this.handleTrain.bind(this);
        this.handleSave = this.handleSave.bind(this);
    }
    componentDidMount() {
        $.getJSON('/api/get_def_parameters')
            .then((params)=>{
                this.setState({data:params.GAN});
            });
    };

    handleSingleParameterChange(value, parameterIdx) {
        let newParameters = this.state.data;
        newParameters[parameterIdx].value = value;
        this.setState({data:newParameters});
    }

    handleTrain(event) {
        train(this.state.data);
    }

    handleSave(event) {
    console.log("hi!")
        $.ajax({
            url: '/api/save_parameters',
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify(this.state.data),
            success: (response) => {
                console.log("hi!")
                console.log(response)
            }
        });
    }

    render() {
        return(
            <div>
                <Grid>
                    <Cell col={12} >
                        <h4> Model Parameters </h4>
                        <ParametersList parameters={this.state.data} onChange={this.handleSingleParameterChange} />
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
        if (!training) {
            view = <div> Tensorboard will show up here</div>;
        } else {

            view =
                <div>
                    <div> realod the page to update Tensorboard</div>
                    <div> you can also navigate to <a href="http://localhost:6006/" target='_blank'>http://localhost:6006/</a> to see this on full screen</div>
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
            15000 // query 4 times in a second
        );
    }

    componentWillUnmount() {
        clearInterval(this.timerID);
    }

    render() {
        return(
            <Grid>
                <Cell col={4} shadow={4}>
                    <TrainingForm isTraining={this.state.isTraining}/>
                </Cell>
                <Cell col={8}>
                    <IframeComponent iframe='iframe' src="http://127.0.0.1:6006" height="100%" width="100%" isTraining={this.state.tensorborad}/>
                </Cell>
            </Grid>
        )
    }
}

function train(inputParameters){
    console.log("train!")
    $.ajax({
        url: '/api/train',
        method: 'POST',
        contentType: 'application/json',
        data: JSON.stringify(inputParameters),
        success: (data) => {
            console.log(data)
        }
    });
}

export default TrainWindow;