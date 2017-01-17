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
import TrainWindow from './TrainWindow';


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