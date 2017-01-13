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

class ParameterSliderRow extends React.Component {
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

class ParameterToggleRow extends React.Component{
    constructor(props) {
        super(props);
        this.handleChange = this.handleChange.bind(this);
    }

    handleChange(e) {
        this.props.onChange(e.target.checked, this.props.idx);
    }
    render() {
        const name = this.props.parameter.name;
        const value = this.props.parameter.value;
        return (
            <tr>
                <td>{name}</td>
                <td>
                    <Switch id={name} checked={value} onChange={this.handleChange}>{value}</Switch>
                </td>
            </tr>
        );
    }
}

class ParameterPathRow extends React.Component{
    constructor(props) {
        super(props);
        this.handleChange = this.handleChange.bind(this);
    }

    handleChange(e) {
        this.props.onChange(e.target.checked, this.props.idx);
    }

    componentDidMount () {
        this._inp.directory = true;
        this._inp.webkitdirectory = true;
    }

    render() {
        const name = this.props.parameter.name;
        const value = this.props.parameter.value;
        const printName = name.replace(/_/g, ' ');
        return (
            <tr>
                <td>Path : {printName}</td>
                <td>
                    <Textfield label="" id={name} value={value} onChange={this.handleChange}/>
                    <div className="mdl-button mdl-button--primary mdl-button--icon mdl-button--file">
                        <i className="material-icons">attach_file</i>
                        <input type="file" id="uploadBtn" ref={i => this._inp = i}/>
                    </div>
                </td>
            </tr>
        );
    }
}


class ParameterTextRow extends React.Component{
    constructor(props) {
        super(props);
        this.handleChange = this.handleChange.bind(this);
    }

    handleChange(e) {
        this.props.onChange(e.target.checked, this.props.idx);
    }
    render() {
        const name = this.props.parameter.name;
        const value = this.props.parameter.value;
        return (
            <tr>
                <td>{name}</td>
                <td>
                    <Textfield label="" id={name} value={value} onChange={this.handleChange}/>
                </td>
            </tr>
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
    }
    render() {
        var rows = [];
        var lastCategory = null;
        let changeFunction = this.handleChange;
        this.props.parameters.forEach(function(parameter, index) {
            let curr_row = null;
            if (parameter.type==='float' || parameter.type==='int') {
                curr_row=<ParameterSliderRow parameter={parameter} key={parameter.name} idx={index} onChange={changeFunction}/>;
            }
            else if (parameter.type==='bool') {
                curr_row=<ParameterToggleRow parameter={parameter} key={parameter.name} idx={index} onChange={changeFunction}/>;
            }
            // else if (parameter.type==='str' && parameter.is_path) {
            //     curr_row=<ParameterPathRow parameter={parameter} key={parameter.name} idx={index} onChange={changeFunction}/>;
            // }
            else if (parameter.type==='str') {
                curr_row=<ParameterTextRow parameter={parameter} key={parameter.name} idx={index} onChange={changeFunction}/>;
            }
            rows.push(curr_row);
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

function getParameters() {
    $.getJSON('/api/get_parameters', function( data ) {
        parameters = data.GAN;
    });
}

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