/* global console */
import { inferenceQuestion } from "./bert/inferenceQuestion";
import { inferenceSentiment } from "./bert/inferenceSentiment";

/**
 * Adds two numbers.
 * @customfunction
 * @param first First number
 * @param second Second number
 * @returns The sum of the two numbers.
 */
export function add(first: number, second: number): number {
  return first + second;
}

/**
 * Returns the sentiment of a string.
 * @customfunction
 * @param text Text string
 * @returns sentiment string.
 */
export async function sentiment(text: string): Promise<string> {
  const result = await inferenceSentiment(text);
  console.log(result[1][0]);
  return result[1][0].toString();
}

/**
 * Returns the sentiment of a string.
 * @customfunction
 * @param question Question string
 * @param context Context string
 * @returns answer string.
 */
export async function question(question: string, context: string): Promise<string> {
  const result = await inferenceQuestion(question, context);
  //
  if (result.length > 0) {
    console.log(result[0].text);
    return result[0].text.toString();
  }
  return "Unable to find answer";
}
