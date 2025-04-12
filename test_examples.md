# Test Examples for FIAS RAG Chat System

This document contains example queries and expected responses for testing the RAG and phishing classification functionality.

## RAG (Analytics) Examples

These examples test the system's ability to retrieve and generate responses based on document content.

### Example 1: Basic Information Retrieval

**Query:**
```
What are the main threats discussed in the cybersecurity documents?
```

**Expected behavior:** The system should search through documents for information about cybersecurity threats and provide a summary based on the retrieved content.

### Example 2: Specific Document Content

**Query:**
```
Can you explain the methodology section from the research paper titled "Analysis of Modern Phishing Techniques"?
```

**Expected behavior:** The system should locate this specific document (if it exists in your database) and provide information specifically from the methodology section.

### Example 3: Statistical Data Request

**Query:**
```
What statistics are provided about phishing attack success rates in the reports?
```

**Expected behavior:** The system should extract numerical data and statistics related to phishing success rates from the documents.

### Example 4: Conceptual Understanding

**Query:**
```
According to the documents, how do spear phishing attacks differ from regular phishing?
```

**Expected behavior:** The system should retrieve and synthesize information that compares these two concepts.

## Phishing Classification Examples

These examples test the system's ability to classify potentially malicious messages.

### Example 1: Clear Phishing Attempt

**Query:**
```
Is this phishing?

Subject: Your account has been suspended
From: accounts@apple-support.securelogin.com

Dear Customer,

We have detected unusual activity on your Apple ID. Your account has been temporarily suspended.
Click here to verify your information: http://apple-verify-account.com/login

Security Team
Apple
```

**Expected behavior:** The system should classify this as phishing with high confidence due to the suspicious URL and urgency.

### Example 2: Legitimate Message

**Query:**
```
Check if this is phishing:

Hello Team,

The quarterly report is now available on the shared drive. Please review it before next Monday's meeting and prepare your feedback.

Thanks,
Sarah Johnson
Marketing Director
```

**Expected behavior:** The system should classify this as not phishing since it contains no suspicious elements.

### Example 3: Ambiguous Case

**Query:**
```
Analyze this for phishing:

Subject: Invoice #INV-2023-456
From: accounting@companyname.com

Please find attached the invoice for last month's services. Payment is due within 15 days.

[attachment: Invoice_2023_456.pdf]
```

**Expected behavior:** This is somewhat ambiguous and could be classified either way, but the confidence score should reflect this uncertainty.

## Mixed Intent Testing

### Example 1: False Classification Request

**Query:**
```
What does the document say about classifying phishing attempts?
```

**Expected behavior:** The system should recognize this as an analytics query about the topic of classification, not a request to classify a specific message.

### Example 2: Complex Classification Request

**Query:**
```
Is this phishing?
I received an email that says: "Your UPS package is on hold. To schedule delivery, click: http://track-ups-package.info/delivery". It seems suspicious to me.
```

**Expected behavior:** The system should recognize this as a classification request and analyze the embedded message.

## Edge Cases

### Example 1: Empty Context

**Query:**
```
What does the document say about quantum computing?
```

**Expected behavior:** If no relevant information exists in the documents, the system should indicate that it doesn't have this information.

### Example 2: Very Long Text for Classification

**Query:**
```
Is this phishing? [followed by several paragraphs of text]
```

**Expected behavior:** The system should handle the long input appropriately, potentially focusing on the most relevant parts for classification.
