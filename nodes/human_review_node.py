def human_review_node(state):
    print("\n [NODE] Human Review")
    
    # Display information to human reviewer
    print("=== HUMAN REVIEW REQUIRED ===")
    print(f"Original Query: {state['input']}")
    print(f"Structured Data: {state['case_data']}")
    
    matching_policies = state.get("matching_policies", [])
    if matching_policies:
        print("Triggered Policies:")
        for policy in matching_policies:
            print(f"  - {policy['policy_id']}: {policy['message']}")
    else:
        print("Triggered Policies: None")
    
    validation_errors = state.get("validation_errors", [])
    if validation_errors:
        print("Reason for Escalation: Validation Errors")
        for error in validation_errors:
            print(f"  - {error}")
    else:
        print("Reason for Escalation: Policy Match or Other")
    
    # Get human decision
    while True:
        response = input("\nHuman Decision (A for Approve, R for Reject): ").strip().upper()
        if response == 'A':
            new_decision = "APPROVED"
            break
        elif response == 'R':
            new_decision = "REJECTED"
            break
        else:
            print("Invalid input. Please enter A or R.")
    
    print(f"Human Decision: {new_decision}")
    
    return {
        "decision": new_decision
    }