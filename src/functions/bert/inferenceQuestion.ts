/* eslint-disable no-undef */
import * as ort from "onnxruntime-web";
import { create_model_input, Feature, getBestAnswers, Answer } from "./question_answer";

export async function inferenceQuestion(question: string, context: string): Promise<Answer[]> {
  const model: string = "./bert-large-uncased-int8.onnx";
  // create session, set options
  const options: ort.InferenceSession.SessionOptions = {
    executionProviders: ["wasm"],
    // executionProviders: ['webgl']
    graphOptimizationLevel: "all",
  };
  console.log("Creating session");
  const session = await ort.InferenceSession.create(model, options);
  //get encoded ids from text tokenizer
  const encoded: Feature = await create_model_input(question, context);
  console.log("encoded", encoded);
  //create input tensor
  const length = encoded.input_ids.length;
  var input_ids = new Array(length);
  var attention_mask = new Array(length);
  var token_type_ids = new Array(length);

  input_ids[0] = BigInt(101);
  attention_mask[0] = BigInt(1);
  token_type_ids[0] = BigInt(0);
  var i = 0;
  for (; i < length; i++) {
    input_ids[i + 1] = BigInt(encoded.input_ids[i]);
    attention_mask[i + 1] = BigInt(1);
    token_type_ids[i + 1] = BigInt(0);
  }
  input_ids[i + 1] = BigInt(102);
  attention_mask[i + 1] = BigInt(1);
  token_type_ids[i + 1] = BigInt(0);

  console.log("arrays", input_ids, attention_mask, token_type_ids);

  const sequence_length = input_ids.length;
  var input_ids_tensor: ort.Tensor = new ort.Tensor("int64", BigInt64Array.from(input_ids), [1, sequence_length]);
  console.log("input_ids_tensor", input_ids_tensor);
  var attention_mask_tensor: ort.Tensor = new ort.Tensor("int64", BigInt64Array.from(attention_mask), [
    1,
    sequence_length,
  ]);
  console.log("attention_mask_tensor", attention_mask_tensor);
  var token_type_ids_tensor: ort.Tensor = new ort.Tensor("int64", BigInt64Array.from(token_type_ids), [
    1,
    sequence_length,
  ]);
  console.log("token_type_ids_tensor", token_type_ids_tensor);

  const model_input: ort.InferenceSession.OnnxValueMapType = {
    input_ids: input_ids_tensor,
    input_mask: attention_mask_tensor,
    segment_ids: token_type_ids_tensor,
  };
  const output_names: ort.InferenceSession.FetchesType = ["start_logits", "end_logits"];
  console.log("model_input", model_input);
  console.log("run session");
  const output = await session.run(model_input, output_names);
  console.log("output", output);
  // loop thru output of string array to bigint array
  const result_length = output["start_logits"].data.length;
  const start_logits: number[] = Array(); //= output["start_logits"].data.map((x: any) => BigInt(x));
  const end_logits: number[] = Array(); //= output["end_logits"].data.map((x: any) => BigInt(x));

  console.log("start_logits", start_logits);
  console.log("end_logits", end_logits);

  for (let i = 0; i <= result_length; i++) {
    start_logits.push(Number(output["start_logits"].data[i]));
  }
  for (let i = 0; i <= result_length; i++) {
    end_logits.push(Number(output["end_logits"].data[i]));
  }

  const answers: Answer[] = getBestAnswers(
    start_logits,
    end_logits,
    encoded.origTokens,
    encoded.tokenToOrigMap,
    context
  );
  console.log("answers", answers);
  return answers;
}
