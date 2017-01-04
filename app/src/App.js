import React, { Component } from "react";
import {
    Button,
    Dialog,
    DialogActions,
    DialogContent,
    DialogTitle,
    FABButton,
    Icon,
    Slider,
    Switch,
    Grid,
    Cell,
    Layout,
    Header,
    Content,
    Textfield
} from "react-mdl";

class ParameterRow extends React.Component {
    constructor(props) {
        super(props);
        this.handleChange = this.handleChange.bind(this);
        this.state = {value: this.props.parameter.value};
    }

    handleChange(value) {
        this.setState({value});
    }

    render() {
        const value = this.state.value;
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

class Parameters extends React.Component {
    render() {
        var rows = [];
        var lastCategory = null;
        console.log(this.props)
        this.props.parameters.forEach(function(parameter) {
            if (parameter.type===1) {
                rows.push(<ParameterRow parameter={parameter} key={parameter.name}/>);
            }
        });
        return (
            <table style={{width: '100%'}}>
                <tbody>{rows}</tbody>
            </table>
        );
    }
}


class ParametersList extends React.Component {
    render() {
        return (
            <div>
                <Parameters parameters={this.props.parameters} />
                <Button raised accent ripple>Train</Button>
            </div>
        );
    }
}

var VEAGAN = [
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
};


class App extends Component {
    render() {
        return (
        <div>
            <Layout fixedHeader>
                <Header title="GooeyBrain" className="mdl-color--teal-800">
                </Header>
                <Content>
                    <Grid className="demo-grid-2">
                        <Cell col={6} shadow={4}>
                            <Grid>
                                <Cell col={12} >
                                    <h4> Model Parameters </h4>
                                    <ParametersList parameters={VEAGAN} />
                                </Cell>
                            </Grid>
                        </Cell>
                        <Cell col={4}>Training Status will show up here</Cell>
                        <Cell col={2}> </Cell>
                    </Grid>
                </Content>
            </Layout>
        </div>
        );
    }
}

export default App;