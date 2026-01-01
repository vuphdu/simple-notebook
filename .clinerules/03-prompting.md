# Image & PDF Prompting (OlmOCR)

Advanced prompting techniques from [OlmOCR](https://github.com/allenai/olmocr) for Vision Language Models.

## Anchor Text Strategy

Provide positional info to ground VLM understanding:

```
Page dimensions: 612.0x792.0
[Image 50x400 to 300x600]
[72x720]Title of Document
[72x680]Author Name
```

Format: `[x_coord x y_coord]text_content`

- Origin at bottom-left (PDF coordinates)
- Images: `[Image x0xy0 to x1xy1]`

---

## Primary Prompt Template

```
Below is the image of one page of a PDF document, as well as some raw textual
content that was previously extracted for it that includes position information
for each image and block of text (The origin [0x0] of the coordinates is in the
lower left corner of the image).

Just return the plain text representation of this document as if you were
reading it naturally.

Turn equations into a LaTeX representation, and tables into markdown format.
Remove the headers and footers, but keep references and footnotes.
Read any natural handwriting.

If there is no text at all, output null.
Do not hallucinate.

RAW_TEXT_START
{anchor_text}
RAW_TEXT_END
```

---

## Structured Output Schema

```json
{
  "primary_language": "en",
  "is_rotation_valid": true,
  "rotation_correction": 0,
  "is_table": false,
  "is_diagram": false,
  "natural_text": "Extracted content..."
}
```

**Key insight**: Field order = model's answer order. Put classification fields BEFORE content!

---

## Best Practices

1. **Use structured outputs** - Consistent results across 1000s of queries
2. **Temperature 0.1** - For document extraction
3. **Check for typos** - Performance difference
4. **Request logprobs** - Debug problematic responses

---

## Math & Tables

**Equations**: Use LaTeX `\( inline \)` and `\[ block \]`

- Avoid Unicode: use `\( \in \)` not `∈`

**Tables**: HTML with `<th>`, `rowspan`, `colspan`

- Don't use `<br>` inside cells

---

## Figure Labeling

```markdown
![Alt text](page_startx_starty_width_height.png)
```

Example:

```markdown
![OAuth2 authentication flowchart](page1_72_200_400_300.png)
```

---

## Image Description Prompt

```
Describe this technical diagram/figure.

Focus on:
1. Diagram type (flowchart, sequence, architecture)
2. Main components/elements
3. Relationships/flows shown
4. Technical concepts illustrated

Include any visible text. Do not hallucinate.
```
