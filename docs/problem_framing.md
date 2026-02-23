# Problem Framing — SMS Spam Baseline Classifier

## Context
The product-trust / content team needs an automated way to distinguish spam SMS
messages from legitimate ("ham") messages. This document translates that need
into specific analytical questions.

---

## Analytical Questions

### 1. What is the baseline spam rate?
What percentage of incoming SMS traffic is spam? Knowing the natural class
balance tells us how hard the problem is and whether a "do-nothing" baseline
(predict everything as ham) is already dangerously accurate.

### 2. Which spam patterns pose the highest risk?
Are there distinct spam sub-types — financial phishing, prize/lottery scams,
adult content, or malware links? Identifying clusters helps the content team
prioritize enforcement actions.

### 3. At what confidence level can we auto-block vs. flag-for-review?
Not all predictions are equally confident. Can we define a high-confidence
threshold where auto-blocking is safe, and a lower threshold where messages
should be routed to human reviewers?

### 4. Which legitimate messages are most likely to be misclassified?
False positives (ham flagged as spam) erode user trust. Which message types
— promotional-sounding ham, messages with URLs, or short ambiguous texts —
are at highest risk of being wrongly blocked?

### 5. How much manual review effort does the classifier save?
Compared to no filter at all, what fraction of the message stream can the
classifier handle automatically, and how many messages still need human eyes?
