// WebGL context for reading raw pixel data of WebP images
let offscreenGlContext = null;

export async function decodeImage(fileBytes, wasm) {
  if (!offscreenGlContext) {
    const canvas = new OffscreenCanvas(1, 1);
    offscreenGlContext = canvas.getContext("webgl2");
    if (!offscreenGlContext) {
      throw new Error("Failed to create WebGL2 context");
    }
  }

  const imageBlob = new Blob([fileBytes]);
  const bitmap = await createImageBitmap(imageBlob, {
    premultiplyAlpha: "none",
  });

  const gl = offscreenGlContext;
  const texture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, bitmap);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);

  const framebuffer = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
  gl.framebufferTexture2D(
    gl.FRAMEBUFFER,
    gl.COLOR_ATTACHMENT0,
    gl.TEXTURE_2D,
    texture,
    0,
  );

  //const data = new Uint8Array(bitmap.width * bitmap.height * 4);
  const len = bitmap.width * bitmap.height * 4;
  const ptr = wasm.exports.alloc_buffer(len);
  const data = new Uint8Array(wasm.exports.memory.buffer, ptr, len);
  gl.readPixels(
    0,
    0,
    bitmap.width,
    bitmap.height,
    gl.RGBA,
    gl.UNSIGNED_BYTE,
    data,
  );

  gl.deleteTexture(texture);
  gl.deleteFramebuffer(framebuffer);

  return { ptr, len, width: bitmap.width, height: bitmap.height };
}
