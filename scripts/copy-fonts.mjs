import { copyFile, mkdir } from 'node:fs/promises';

const destination = new URL('../static/fonts/', import.meta.url);
await mkdir(destination, { recursive: true });
for (const font of ['playfair-display', 'source-serif-4', 'jetbrains-mono']) {
  const source = new URL(`../node_modules/@fontsource-variable/${font}/`, import.meta.url);
  await copyFile(new URL(`files/${font}-latin-wght-normal.woff2`, source), new URL(`${font}-latin-wght-normal.woff2`, destination));
  await copyFile(new URL('LICENSE', source), new URL(`${font}-LICENSE.txt`, destination));
}
