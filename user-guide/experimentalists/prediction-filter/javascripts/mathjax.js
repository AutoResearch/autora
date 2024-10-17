// required for equation display in documentation
window.MathJax = {
    tex: {
      inlineMath: [ ["\\(","\\)"] ],
      displayMath: [ ["\\[","\\]"] ],
      processEscapes: true,
      processEnvironments: true
    },
    options: {
      ignoreHtmlClass: ".*",
      processHtmlClass: "arithmatex"
    }
  };