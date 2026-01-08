# Semantic Router Training Data

Copy and paste these into the Designer UI to train your semantic router.

## Billing Route

**Route Name:** billing
**Target Model:** billing_specialist
**Description:** Questions about bills, payments, invoices, and account balances

### Utterances (copy all):
```
what is my bill
how much do I owe
payment options
invoice question
when is my payment due
can I pay in installments
billing statement
account balance
credit card payment
autopay setup
where can I see my charges
why was I charged twice
refund my payment
update payment method
payment failed
```

---

## Support Route

**Route Name:** support
**Target Model:** tech_support
**Description:** Technical support, login issues, and account problems

### Utterances (copy all):
```
help with login
password reset
can't access my account
technical problem
app not working
connection issues
how do I change my settings
two-factor authentication
app keeps crashing
error message
forgot my username
account locked
session expired
can't verify my email
slow loading
```

---

## Sales Route

**Route Name:** sales
**Target Model:** sales_team
**Description:** Pricing, plans, upgrades, and sales inquiries

### Utterances (copy all):
```
pricing information
enterprise plan
get a quote
upgrade subscription
product features
team plan
annual discount
compare plans
free trial
cancel subscription
what's included
volume pricing
custom solution
demo request
contract terms
```

---

## Router Settings

| Setting | Value |
|---------|-------|
| Router Name | customer_support_router |
| Embedder Model | sentence-transformers/all-MiniLM-L6-v2 (Recommended) |
| Default Model | general_assistant |
| Similarity Threshold | 0.6 |

---

## Quick Setup Instructions

1. Open the Designer UI at http://localhost:3000
2. Navigate to Models > Train > Semantic Router
3. For each route above:
   - Click "Add route"
   - Fill in the Route Name and Target Model
   - Add the Description (used for AI-generated utterances)
   - Paste the utterances (one per line)
4. Set the Router Settings as shown above
5. Click "Train Router"
6. Test with sample queries in the Test section
