/* eslint-disable no-undef */
import * as ort from "onnxruntime-web";

export async function create_model_input(encoded: number[]) {
  var input_ids = new Array(encoded.length + 2);
  var attention_mask = new Array(encoded.length + 2);
  var token_type_ids = new Array(encoded.length + 2);
  input_ids[0] = BigInt(101);
  attention_mask[0] = BigInt(1);
  token_type_ids[0] = BigInt(0);
  var i = 0;
  for (; i < encoded.length; i++) {
    input_ids[i + 1] = BigInt(encoded[i]);
    attention_mask[i + 1] = BigInt(1);
    token_type_ids[i + 1] = BigInt(0);
  }
  input_ids[i + 1] = BigInt(102);
  attention_mask[i + 1] = BigInt(1);
  token_type_ids[i + 1] = BigInt(0);
  const sequence_length = input_ids.length;
  var input_ids_tensor = new ort.Tensor("int64", BigInt64Array.from(input_ids), [1, sequence_length]);
  var attention_mask_tensor = new ort.Tensor("int64", BigInt64Array.from(attention_mask), [1, sequence_length]);
  var token_type_ids_tensor = new ort.Tensor("int64", BigInt64Array.from(token_type_ids), [1, sequence_length]);
  return {
    input_ids: input_ids_tensor,
    attention_mask: attention_mask_tensor,
    token_type_ids: token_type_ids_tensor,
  };
}

export function sortResult(a, b) {
  if (a[1] === b[1]) {
    return 0;
  } else {
    return a[1] < b[1] ? 1 : -1;
  }
}

export function sigmoid(t) {
  return 1 / (1 + Math.pow(Math.E, -t));
}
