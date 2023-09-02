const chai = require('chai');
const fs = require('fs');

const wasm_tester = require('circom_tester').wasm;

const F1Field = require('ffjavascript').F1Field;
const Scalar = require('ffjavascript').Scalar;
exports.p = Scalar.fromString('21888242871839275222246405745257275088548364400416034343698204186575808495617');
const Fr = new F1Field(exports.p);

const assert = chai.assert;

const exec = require('await-exec');

const best_practice = require('../models/best_practice.json');
const alt_model = require('../models/alt_model.json');

function softmax(arr) {
    return arr.map(function(value,index) { 
      return Math.exp(value) / arr.map( function(y /*value*/){ return Math.exp(y) } ).reduce( function(a,b){ return a+b })
    })
}

describe('torch2circom test', function () {
    this.timeout(100000000);

    describe('transformer', async () => {
        it('raw output', async () => {
            // await exec('python3 main.py models/best_practice.h5 -o best_practice_raw --raw');

            let INPUT = JSON.parse(fs.readFileSync('./transformer/transformer_input_scale9.json'));
            let OUTPUT = JSON.parse(fs.readFileSync('./transformer/transformer_output.json'));

            // let INPUT = {};
            // for (const [key, value] of Object.entries(json)) {
            //     if (Array.isArray(value)) {
            //         let tmpArray = [];
            //         for (let i = 0; i < value.flat().length; i++) {
            //             tmpArray.push(Fr.e(value.flat()[i]));
            //         }
            //         INPUT[key] = tmpArray;
            //     } else {
            //         INPUT[key] = Fr.e(value);
            //     }
            // }

            const circuit = await wasm_tester('./transformer_circuit.circom');
            
            const witness = await circuit.calculateWitness(INPUT, true);
            
            assert(Fr.eq(Fr.e(witness[0]),Fr.e(1)));

            const scale = 1E-77;

            let predicted = [];
            for (var i=0; i<OUTPUT['out'].length; i++) {
                predicted.push(parseFloat(Fr.toString(Fr.e(witness[i+1]))) * scale);
            }

            let ape = 0;

            for (var i=0; i<OUTPUT['out'].length; i++) {
                const actual = OUTPUT['out'][i];
                console.log('actual', actual, 'predicted', predicted[i]);
                ape += Math.abs((predicted[i]-actual)/actual);
            }

            const mape = 100*ape/OUTPUT['out'].length;

            console.log('mean absolute error %', mape);
        });
    });
});