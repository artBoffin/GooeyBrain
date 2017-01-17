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
                <td></td>
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
        this.props.onChange(e.target.value, this.props.idx);
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
                <td></td>
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
        this.props.onChange(e.target.value, this.props.idx);
    }
    render() {
        const name = this.props.parameter.name;
        const value = this.props.parameter.value;
        return (
            <tr>
                <td>{name}</td>
                <td>
                    <Textfield label="" id={name} value={value} style={{width: '100%'}} onChange={this.handleChange}/>
                </td>
                <td></td>
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

export default ParametersList;