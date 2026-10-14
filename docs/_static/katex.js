// Render the math that pymdownx.arithmatex wraps in `.arithmatex` elements.
document$.subscribe(({ body }) => {
  for (const element of body.querySelectorAll(".arithmatex")) {
    renderMathInElement(element, {
      delimiters: [
        { left: "\\(", right: "\\)", display: false },
        { left: "\\[", right: "\\]", display: true },
      ],
    });
  }
});
