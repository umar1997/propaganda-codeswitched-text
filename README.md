# Detecting Propaganda Techniques in Code-Switched Social Media Text (EMNLP'23)

> [**Detecting Propaganda Techniques in Code-Switched Social Media Text**](https://arxiv.org/abs/2305.14534)<br>
> [Muhammad Umar Salman](https://scholar.google.com/citations?user=vR7aZzAAAAAJ&hl=en&oi=sra),
[Asif Hanif](https://scholar.google.com/citations?hl=en&user=6SO2wqUAAAAJ),
[Shady Shehata](https://scholar.google.com/citations?hl=en&user=osOiYvYAAAAJ)
and [Preslav Nakov](https://scholar.google.com/citations?hl=en&user=DfXsKZ4AAAAJ)

[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2305.14534)
[![poster](https://img.shields.io/badge/Presentation-Poster-F9D371)](https://github.com/umar1997/propaganda-codeswitched-text/tree/main/Media/Poster.pdf)
[![slides](https://img.shields.io/badge/Presentation-Slides-B762C1)](https://github.com/umar1997/propaganda-codeswitched-text/tree/main/Media/Slides.pdf)


<hr />

| ![main figure](/Media/Images/main_figure.png)|
|:--| 



> **Abstract:** <p style="text-align: justify;">*Propaganda is a form of communication intended to influence the opinions and the mindset of the public to promote a particular agenda. With the rise of social media, propaganda has spread rapidly, leading to the need for automatic propaganda detection systems. Most work on propaganda detection has focused on high-resource languages, such as English, and little effort has been made to detect propaganda for low-resource languages. Yet, it is common to find a mix of multiple languages in social media communication, a phenomenon known as code-switching. Code-switching combines different languages within the same text, which poses a challenge for automatic systems. Considering this premise, we propose a novel task of detecting propaganda techniques in code-switched text. To support this task, we create a corpus of 1,030 texts code-switching between English and Roman Urdu, annotated with 20 propaganda techniques at fragment-level. We perform a number of experiments contrasting different experimental setups, and we find that it is important to model the multilinguality directly rather than using translation as well as to use the right fine-tuning strategy.* </p>
<hr />


## Contributions
1. **Formulation of Novel NLP Task:** We formulate the novel NLP task of detecting propaganda techniques in code-switched text in the languages (English and Roman Urdu)
2. **Creation of Annotated Corpus:** We construct and annotate a new corpus specifically for this task, comprising 1,030 code-switched texts in English and Roman Urdu. These texts are annotated at a fragmentlevel with 20 propaganda techniques.
3. **Evaluating different NLP Models:** We experiment with various model classes, including monolingual, multilingual, crosslingual models, and Large Language Models (LLMs), for this task and dataset and we provide a comparative performance analysis.
4. **Developed a Web-based Platform:** We design and create a new website platform with a user interface to  annotate spans of text and label them as different propaganda techniques.


<hr />

## Contact
Should you have any question, please create an issue on this repository or contact at **umar.salman@mbzuai.ac.ae**

<hr />