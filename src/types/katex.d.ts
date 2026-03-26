declare module "katex" {
  export function renderToString(
    formula: string,
    options?: {
      displayMode?: boolean;
      throwOnError?: boolean;
    },
  ): string;
}
