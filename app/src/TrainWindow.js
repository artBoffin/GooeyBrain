import React, { Component } from "react";
import {
    Button,
    Grid,
    Cell
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
                this.setState({data:params.dcgan_model});
            });
    };

    handleSingleParameterChange(value, parameterIdx) {
        let newParameters = this.state.data;
        newParameters[parameterIdx].value = value;
        this.setState({data:newParameters});
    }

    handleTrain(event) {
        train(this.state.data);
        this.props.setTrain();
    }

    handleSave(event) {
        $.ajax({
            url: '/api/save_parameters',
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify(this.state.data),
            success: (response) => {
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

class TrainLog extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            logMessages: [],
        };
    }

    openLink() {
        const shell = window.require('electron').shell;
        shell.openExternal('http://127.0.0.1:6006/')
    }

    componentDidMount() {
        let component = this
        const ipc = window.require('electron').ipcRenderer;
        ipc.on('info-log', function (event, data) {
            var logMessages = component.state.logMessages.slice()
            logMessages.push(data.msg)
            component.setState({logMessages:logMessages})
        });
    }

    render() {
        const training = this.props.isTraining;
        let view = null;
        const n_messages = 20
        if (!training) {
            view = <div> Training logs will show up here</div>;
        } else {
            let logRows = [];
            this.state.logMessages.forEach(function(msg, index) {
                logRows.push(<div key={index}>{msg}</div>);
            });
            view =
                <div>
                <div> Navigate to <a onClick={this.openLink}>http://localhost:6006/</a> to see Tensorboard logs</div>
                <div>
                    <h4> logging messages:</h4>
                    {logRows}
                </div>
                </div>
        }
        return (
            <div>
                {view}
            </div>
        )
    }
}

class TrainWindow extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            isTraining: false,
        };
        this.setTrainingStatus = this.setTrainingStatus.bind(this);
    }

    setTrainingStatus() {
        console.log('set training called');
        this.setState({isTraining:true});
    }

    render() {
        return(
            <Grid>
                <Cell col={4} shadow={4}>
                    <TrainingForm isTraining={this.state.isTraining} setTrain={this.setTrainingStatus}/>
                </Cell>
                <Cell col={8}>
                    <TrainLog isTraining={this.state.isTraining}/>
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