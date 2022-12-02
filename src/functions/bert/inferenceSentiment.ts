/* eslint-disable no-undef */
import * as bertProcessing from "./bertProcessing";
import * as ort from "onnxruntime-web";
import { EMOJIS } from "./emoji";
import { loadTokenizer } from "./bert_tokenizer";

export async function inferenceSentiment(text: string) {
  const model: string = "./xtremedistill-go-emotion-int8.onnx";
  // create session, set options
  const options: ort.InferenceSession.SessionOptions = {
    executionProviders: ["wasm"],
    // executionProviders: ['webgl']
    graphOptimizationLevel: "all",
  };
  console.log("Creating session");
  const session = await ort.InferenceSession.create(model, options);
  //get encoded ids from text tokenizer
  const tokenizer = loadTokenizer();
  const encoded = await tokenizer.then((t) => {
    return t.tokenize(text);
  });
  console.log("encoded", encoded);
  const model_input = await bertProcessing.create_model_input(encoded);
  console.log("model input", model_input);
  console.log("run session");
  const output = await session.run(model_input, ["output_0"]);
  console.log("output", output);

  const outputResult = output["output_0"].data;
  console.log("outputResult", outputResult);

  let probs = [];
  for (let i = 0; i < outputResult.length; i++) {
    let sig = bertProcessing.sigmoid(outputResult[i]);
    probs.push(Math.floor(sig * 100));
  }

  console.log("probs", probs);
  const result = [];
  for (var i = 0; i < EMOJIS.length; i++) {
    const t = [EMOJIS[i], probs[i]];
    result[i] = t;
  }
  result.sort(bertProcessing.sortResult);
  console.log(result);
  const result_list = [];
  result_list[0] = ["Emotion", "Score"];
  for (i = 0; i < 6; i++) {
    result_list[i + 1] = result[i];
  }
  console.log(result_list);
  return result_list;
}
