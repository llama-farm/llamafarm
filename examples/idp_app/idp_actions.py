from sdk import tool
import json
import time

@tool(description="Process an invoice payment.")
def pay_invoice(vendor: str, amount: str, invoice_number: str) -> str:
    """
    Executes a payment transaction for a verified invoice.
    """
    print(f"\n[ACTION] Processing Payment...")
    print(f"  > Vendor: {vendor}")
    print(f"  > Amount: {amount}")
    print(f"  > Ref:    {invoice_number}")
    time.sleep(1) # Simulate API call
    return json.dumps({
        "status": "success",
        "transaction_id": f"TXN-{int(time.time())}",
        "message": f"Payment of {amount} to {vendor} initiated."
    })

@tool(description="Archive a legal document securely.")
def archive_document(doc_type: str, parties: str, expiry_date: str) -> str:
    """
    Encrypts and archives a legal document like an NDA or Contract.
    """
    print(f"\n[ACTION] Archiving Document...")
    print(f"  > Type:    {doc_type}")
    print(f"  > Parties: {parties}")
    print(f"  > Expiry:  {expiry_date}")
    time.sleep(1) # Simulate Database Op
    return json.dumps({
        "status": "archived",
        "doc_id": f"DOC-{int(time.time())}",
        "retention_policy": "7_YEARS"
    })

@tool(description="Flag an anomaly or unknown document for human review.")
def flag_anomaly(reason: str, confidence_score: float) -> str:
    """
    Flags content that cannot be automatically processed or looks suspicious.
    """
    print(f"\n[ACTION] flagging Anomaly...")
    print(f"  > Reason: {reason}")
    print(f"  > Score:  {confidence_score}")
    return json.dumps({
        "status": "flagged_for_review",
        "ticket_id": f"TKT-{int(time.time())}"
    })
