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
            all_data: [],
            data:[],
            selected_model: "",
            models:[],
            isTraining: training
        };

        this.handleSingleParameterChange = this.handleSingleParameterChange.bind(this);
        this.handleTrain = this.handleTrain.bind(this);
        this.handleSave = this.handleSave.bind(this);
        this.handleLoad = this.handleLoad.bind(this);
    }
    componentDidMount() {
        $.getJSON('/api/get_def_parameters')
            .then((params)=>{
                var selected = params.selected_model;
                let models = Object.keys(params).slice()
                models.pop("selected_model")
                console.log(selected)
                var data = params[selected].slice()
                this.setState({
                    all_data:params,
                    data:data,
                    selected_model:selected,
                    models:models
                });
                console.log("Models: "+this.state.models);
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
        // Download the parameter as file
        var dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(this.state.data));
        var dlAnchorElem = document.getElementById('downloadAnchorElem');
        dlAnchorElem.setAttribute("href",     dataStr     );
        dlAnchorElem.setAttribute("download", "parameters.json");
        dlAnchorElem.click();
    }

    handleLoad(event) {
        const fs = require('fs');
        const ipc = window.require('electron').ipcRenderer;
        ipc.send('open-file-dialog');
        const call=this.handleChange;
        const component = this;
        ipc.on('selected-file', function (event, path) {
            let filepath = path[0]
            fs.openSync(filepath, 'r+'); //throws error if file doesn't exist
            var data=fs.readFileSync(filepath); //file exists, get the contents
            let parsedParameters = JSON.parse(data); //turn to js object
            component.setState({data:parsedParameters})
        })
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
                </Grid>
                <Grid>
                    <Cell col={6} >
                        <Button raised ripple onClick={this.handleLoad}>Load Parameters</Button>
                    </Cell>
                    <Cell col={6} >
                        <Button raised ripple onClick={this.handleSave}>Save Parameters</Button>
                        <a id="downloadAnchorElem" style={{display:'none'}}></a>
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