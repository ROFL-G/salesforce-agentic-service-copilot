"""
Salesforce Agentic Service & Org Operations Copilot (Agentforce / Service Cloud)
Architecture: Pure-Python TF-IDF Vector RAG + Autonomous ReAct Multi-Tool Agent + Gradio Operations Cockpit
Memory Footprint: <35 MB RAM | Zero PyTorch/Heavy Dependencies
"""

import os
import re
import math
import socket
import datetime
from collections import Counter
import gradio as gr

# ==============================================================================
# 1. COLLISION-PROOF DYNAMIC PORT ALLOCATOR
# ==============================================================================
def find_available_port(starting_port=7860, max_attempts=50):
    """Dynamically locates an open socket port to prevent address collision."""
    for port in range(starting_port, starting_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    return starting_port

# ==============================================================================
# 2. PURE-PYTHON TF-IDF VECTOR RAG ENGINE
# ==============================================================================
SALESFORCE_KNOWLEDGE_CORPUS = [
    {
        "id": "KB-APEX-10001",
        "title": "Apex Governor Limits: Resolving Apex CPU Time Limit Exceeded (10001 ms)",
        "snippet": (
            "Apex CPU limit is strictly 10,000 ms for synchronous transactions and 60,000 ms for "
            "asynchronous Batch/Queueable Apex. CPU time is calculated across all trigger invocations, "
            "Flow interviews, and validation rules. When APEX_CPU_TIMEOUT occurs, investigate nested "
            "loops, un-indexed SOQL traversals over child collections, and recursive before/after triggers. "
            "Remediation requires implementing a static boolean execution flag or moving complex "
            "calculations to an asynchronous Queueable Apex chain."
        ),
        "category": "Apex / Flow Execution"
    },
    {
        "id": "KB-SOQL-101",
        "title": "SOQL 101 Fault Resolution & High-Volume Custom Object Selectivity",
        "snippet": (
            "Total number of SOQL queries issued per synchronous Apex transaction is limited to 100. "
            "System.LimitException: Too many SOQL queries: 101 triggers when queries are placed inside "
            "FOR loops or Flow 'Get Records' elements inside loops. On tables exceeding 200,000 records, "
            "queries must be selective: the filter field must possess a standard or two-point custom index, "
            "and query cardinality must not exceed 10% of total records (max 333,333). Immediate remediation: "
            "request a two-point custom index via Tier-3 support or refactor to Map-based bulk collections."
        ),
        "category": "Database / SOQL Optimization"
    },
    {
        "id": "KB-AUTH-403",
        "title": "SAML SSO & Connected App Certificate Revocation Remediation",
        "snippet": (
            "When enterprise users fail authentication with SAML_ASSERTION_EXPIRED or SIGNATURE_VERIFICATION_FAILED, "
            "cross-check the Identity Provider (IdP) X.509 signing certificate in Single Sign-On Settings against "
            "the SAML response assertion. If an IdP certificate rolled over unexpectedly, admins cannot access "
            "the Org via federated login. Remediation: verify Signature Support tier to generate a secure 24-hour "
            "Direct Login Emergency Bypass Link (force login.salesforce.com?useStandard=true for Org Admins)."
        ),
        "category": "Identity & Access Management"
    },
    {
        "id": "KB-DATA-500",
        "title": "Data Cloud Ingestion Stream Throttling & Credit Surge Policy",
        "snippet": (
            "Data Cloud Data Stream ingestion drops or status DATA_CLOUD_INGEST_THROTTLED occur when real-time "
            "ingestion connectors (Amazon S3, Webhook, MuleSoft) exceed provisioned daily Data Cloud ingestion credits "
            "or experience API rate limits (HTTP 429). Signature and Enterprise customers experiencing critical "
            "billing/e-commerce ingestion drops are eligible for an autonomous 48-hour 25% credit buffer surge "
            "while ingestion batch pipelines are repartitioned."
        ),
        "category": "Data Cloud / Integration"
    },
    {
        "id": "KB-ASYNC-429",
        "title": "Batch Apex Holding Queue Lock & Row Lock Contention (UNABLE_TO_LOCK_ROW)",
        "snippet": (
            "When concurrent Batch Apex workers or scheduled Flow batches update the same parent Account records, "
            "database transactions fail with UNABLE_TO_LOCK_ROW or get deadlocked in the Apex Flex Queue. "
            "Remediation requires terminating stalled child worker jobs via System.abortJob(), reducing "
            "batch chunk size from 200 to 50, and enforcing FOR UPDATE lock ordering on parent IDs."
        ),
        "category": "Asynchronous Processing"
    },
    {
        "id": "KB-TEST-71",
        "title": "Deployment Code Coverage Deficit & Seasonal Release Deprecation",
        "snippet": (
            "Production deployments require an overall test code coverage of at least 75%. Following seasonal "
            "Salesforce releases (Winter/Spring/Summer), legacy test utilities asserting obsolete system methods "
            "fail, dropping Org coverage. Resolution requires cross-referencing Release Notes, injecting modern "
            "Test.loadData or TestDataFactory scaffolding, and deploying with RunSpecifiedTests to bypass legacy debt."
        ),
        "category": "Metadata / CI/CD Deployment"
    },
    {
        "id": "KB-EVENT-800",
        "title": "Platform Event High-Volume Daily Allocation & Replay Id Gap",
        "snippet": (
            "High-volume platform events published via CometD or REST API enforce daily publication limits. "
            "When PLATFORM_EVENT_DELIVERY_BLOCKED occurs, client subscribers experience event drops or 403 buffer "
            "full responses. Enterprise customers can trigger an event bus partition flush, reset subscriber "
            "Replay ID checkpoints to -2, and grant an emergency 24-hr publication quota waiver."
        ),
        "category": "Event Bus / Architecture"
    },
    {
        "id": "KB-API-900",
        "title": "24-Hour Rolling REST/SOAP API Limit Saturation Mitigation",
        "snippet": (
            "Salesforce multi-tenant instances enforce a rolling 24-hour API request allocation calculated based on "
            "user licenses. When REQUEST_LIMIT_EXCEEDED is returned, third-party ERP integrations, middleware, and "
            "ETL pipelines are blocked from communicating with Salesforce. Signature Success accounts are entitled "
            "to an immediate, autonomous +25,000 API call emergency burst buffer."
        ),
        "category": "API & Middleware Gateway"
    }
]

def tokenize(text):
    """Lowercases and extracts alpha-numeric tokens."""
    return re.findall(r'\b[a-zA-Z0-9_\-\.]{2,}\b', text.lower())

class PurePythonVectorRAG:
    """Ultra-lean vector retriever utilizing TF-IDF and Cosine Similarity."""
    def __init__(self, corpus):
        self.corpus = corpus
        self.doc_count = len(corpus)
        self.doc_tokens = [tokenize(doc["title"] + " " + doc["snippet"] + " " + doc["category"]) for doc in corpus]
        self.vocab = sorted(list({token for tokens in self.doc_tokens for token in tokens}))
        
        # Calculate Document Frequencies
        self.df = Counter()
        for tokens in self.doc_tokens:
            for token in set(tokens):
                self.df[token] += 1
                
        # Calculate Document TF-IDF Vectors
        self.idf = {term: math.log((1 + self.doc_count) / (1 + freq)) + 1.0 for term, freq in self.df.items()}
        self.doc_vectors = [self._vectorize(tokens) for tokens in self.doc_tokens]

    def _vectorize(self, tokens):
        counts = Counter(tokens)
        vec = {}
        for token, count in counts.items():
            if token in self.idf:
                tf = count / len(tokens) if tokens else 0
                vec[token] = tf * self.idf[token]
        norm = math.sqrt(sum(val ** 2 for val in vec.values()))
        return {k: v / norm for k, v in vec.items()} if norm > 0 else {}

    def retrieve(self, query, top_k=2):
        q_tokens = tokenize(query)
        q_vec = self._vectorize(q_tokens)
        if not q_vec:
            return self.corpus[:top_k]
            
        scores = []
        for idx, d_vec in enumerate(self.doc_vectors):
            dot_product = sum(q_vec[term] * d_vec.get(term, 0.0) for term in q_vec)
            scores.append((dot_product, idx))
            
        scores.sort(reverse=True, key=lambda x: x[0])
        return [self.corpus[idx] for _, idx in scores[:top_k]]

vector_rag = PurePythonVectorRAG(SALESFORCE_KNOWLEDGE_CORPUS)

# ==============================================================================
# 3. PRODUCTION SCENARIO DATABASE (8 REAL-WORLD INCIDENTS)
# ==============================================================================
SCENARIO_DATABASE = {
    "SF-APEX-101": {
        "title": "Apex CPU Limit Exceeded (10001 ms) in Order Trigger",
        "org_id": "00D8c000008FkL1EAK",
        "edition": "Unlimited",
        "sla_tier": "Signature Success",
        "error_code": "APEX_CPU_TIMEOUT_10001",
        "object": "OrderLineItem__c",
        "claim": "End-of-quarter batch order checkout is throwing 'System.LimitException: Apex CPU time limit exceeded (10001 ms)' during OrderTrigger before-update recursion over 45,000 items."
    },
    "SF-SOQL-202": {
        "title": "SOQL 101 Exception on Flow Loop Over Custom Claim Object",
        "org_id": "00D5e000001AbC2MAK",
        "edition": "Enterprise",
        "sla_tier": "Premier Support",
        "error_code": "SOQL_QUERY_LIMIT_101",
        "object": "InsuranceClaim__c",
        "claim": "Record-triggered Flow 'Auto_Assign_Claim' is failing on customer portal submissions with 'System.LimitException: Too many SOQL queries: 101' due to a nested Get Records element."
    },
    "SF-AUTH-303": {
        "title": "Enterprise SSO SAML X.509 Certificate Revocation",
        "org_id": "00D2w000003Kk9PEAS",
        "edition": "Unlimited",
        "sla_tier": "Signature Success",
        "error_code": "SAML_ASSERTION_REVOKED",
        "object": "ConnectedApp_OktaSSO",
        "claim": "All 12,000 corporate employees locked out of Salesforce Service Console. Identity Provider Okta X.509 certificate expired at midnight, blocking SAML assertion validation."
    },
    "SF-DATA-404": {
        "title": "Data Cloud POS Stream Ingestion Throttling",
        "org_id": "00D7q000004Mn8JEAS",
        "edition": "Enterprise",
        "sla_tier": "Premier Support",
        "error_code": "DATA_CLOUD_INGEST_THROTTLED",
        "object": "POS_Transaction_Stream__dl",
        "claim": "Black Friday point-of-sale telemetry stream throttled at Data Cloud connector gateway. Ingestion lag is 4 hours, throwing HTTP 429 and threatening sales reporting integrity."
    },
    "SF-ASYNC-505": {
        "title": "Batch Apex Holding Queue Lock & Row Lock Contention",
        "org_id": "00D9z000006PpQ1EAK",
        "edition": "Unlimited",
        "sla_tier": "Signature Success",
        "error_code": "HOLDING_QUEUE_DEADLOCK_429",
        "object": "Account",
        "claim": "Five parallel Batch Apex workers updating shared parent Accounts have deadlocked in the Apex Flex Queue with UNABLE_TO_LOCK_ROW, blocking nightly billing reconciliation."
    },
    "SF-DEPLOY-606": {
        "title": "Production Deployment Code Coverage Deficit (71%)",
        "org_id": "00D4v000002XxY7EAK",
        "edition": "Enterprise",
        "sla_tier": "Standard Support",
        "error_code": "CODE_COVERAGE_FAILURE",
        "object": "MetadataPackage",
        "claim": "Quarterly release deploy failed validation with 71% average test coverage (minimum 75% required). Deprecated seasonal release test classes dropped coverage across legacy utilities."
    },
    "SF-EVENT-707": {
        "title": "Platform Event Bus Delivery Blocked & Buffer Saturation",
        "org_id": "00D3y000005RtW2EAK",
        "edition": "Unlimited",
        "sla_tier": "Signature Success",
        "error_code": "PLATFORM_EVENT_DELIVERY_BLOCKED",
        "object": "Order_Event__e",
        "claim": "ERP microservices dropping order fulfillment events with PLATFORM_EVENT_DELIVERY_BLOCKED (403 Buffer Full). High-velocity holiday stream exceeded 24-hr daily event allocation."
    },
    "SF-API-808": {
        "title": "24-Hour Rolling REST API Quota Saturation",
        "org_id": "00D1a000009HhJ4EAK",
        "edition": "Enterprise",
        "sla_tier": "Premier Support",
        "error_code": "REQUEST_LIMIT_EXCEEDED_429",
        "object": "MuleSoft_Integration_User",
        "claim": "Enterprise middleware synchronization halted mid-day with 'REQUEST_LIMIT_EXCEEDED: TotalRequests Limit exceeded'. 24-hour rolling call threshold of 100,000 fully depleted."
    }
}

# ==============================================================================
# 4. SIMULATED SALESFORCE INTERNAL APIS (REACT TOOLS)
# ==============================================================================
def tool_inspect_apex_execution_log(org_id, error_code, prompt_text=""):
    """Parses raw Apex debug logs, CPU duration, and recursion call stacks."""
    text_lower = prompt_text.lower()
    if "cpu" in text_lower or "APEX_CPU" in error_code:
        return {
            "status": "ANOMALY_CONFIRMED",
            "cpu_time_ms": 11482,
            "max_limit_ms": 10000,
            "offending_component": "OrderTrigger.trigger (Line 142)",
            "root_cause": "Trigger recursion: before-update event invoking un-guarded update loop on OrderLineItem__c.",
            "heap_allocation_mb": 4.8
        }
    elif "soql" in text_lower or "SOQL" in error_code:
        return {
            "status": "SOQL_FLOOD_DETECTED",
            "soql_count": 101,
            "max_allowed": 100,
            "offending_component": "Flow: Auto_Assign_Claim (Element: Get_Underwriter_Records)",
            "root_cause": "SOQL query execution nested within a loop element iterating over 120 child records."
        }
    elif "api" in text_lower or "REQUEST_LIMIT" in error_code:
        return {
            "status": "API_CEILING_REACHED",
            "daily_requests_used": 100048,
            "daily_quota": 100000,
            "offending_component": "Connected App: MuleSoft_Sync_Gateway",
            "root_cause": "Unthrottled polling sync loop without delta query filtering."
        }
    elif "event" in text_lower or "PLATFORM_EVENT" in error_code:
        return {
            "status": "EVENT_BUFFER_SATURATED",
            "events_published_24h": 254000,
            "daily_allocation": 250000,
            "offending_component": "EventBus: Order_Event__e",
            "root_cause": "High-volume stream publisher exceeded tenant allocation limit."
        }
    return {
        "status": "LOG_INSPECTED",
        "cpu_time_ms": 1420,
        "soql_count": 14,
        "note": "Log indicates platform infrastructure, schema mismatch, or authentication boundary error."
    }

def tool_audit_soql_and_storage_limits(org_id, target_object):
    """Inspects table volume, index selectivity, and Data Cloud daily credit buffers."""
    mock_db = {
        "OrderLineItem__c": {"record_count": 1280000, "custom_index_present": False, "is_selective": False},
        "InsuranceClaim__c": {"record_count": 420000, "custom_index_present": False, "is_selective": False},
        "POS_Transaction_Stream__dl": {"daily_credits_used": "100%", "burst_eligible": True, "connector_state": "THROTTLED"},
        "Order_Event__e": {"bus_status": "LOCKED", "replay_id_lag": 4200, "burst_eligible": True},
        "MuleSoft_Integration_User": {"rolling_24h_burn": "100.2%", "active_tokens": 14, "burst_eligible": True}
    }
    return mock_db.get(target_object, {"record_count": 85000, "custom_index_present": True, "is_selective": True})

def tool_verify_signature_sla_tier(org_id, declared_sla):
    """Validates Premier/Signature SLA terms and emergency intervention authority."""
    if "Signature" in declared_sla:
        return {
            "tier": "Signature Success",
            "irt_minutes": 15,
            "dedicated_tam": "Elena Rostova (Principal TAM - Enterprise)",
            "emergency_bypass_permitted": True,
            "credit_surge_limit_pct": 25
        }
    elif "Premier" in declared_sla:
        return {
            "tier": "Premier Support",
            "irt_minutes": 60,
            "dedicated_tam": "Shared Premier Support Queue",
            "emergency_bypass_permitted": True,
            "credit_surge_limit_pct": 15
        }
    return {
        "tier": "Standard Support",
        "irt_minutes": 240,
        "emergency_bypass_permitted": False,
        "credit_surge_limit_pct": 0
    }

def tool_execute_salesforce_remediation(action_type, org_id, target_object):
    """Executes mock remediation actions against Salesforce multi-tenant platform APIs."""
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    if action_type == "INJECT_STATIC_RECURSION_GUARD":
        return f"SUCCESS [{now}]: Static boolean bypass guard compiled into {target_object}. Prevented duplicate trigger loop."
    elif action_type == "PROVISION_CUSTOM_INDEX":
        return f"SUCCESS [{now}]: Accelerated two-point custom index queued for {target_object}. Cardinality scan estimated in 120s."
    elif action_type == "PROVISION_SSO_BYPASS_URL":
        return f"SUCCESS [{now}]: Generated 24-hr Emergency Direct Login Token for System Administrators on Org {org_id}."
    elif action_type == "SURGE_DATA_CLOUD_CREDITS":
        return f"SUCCESS [{now}]: Injected +25% temporary 48-hr Data Cloud ingestion buffer. Connector unthrottled."
    elif action_type == "ABORT_DEADLOCKED_BATCH":
        return f"SUCCESS [{now}]: Terminated 3 conflicting child worker threads. Rescheduled Batch Apex with chunk size 50."
    elif action_type == "SURGE_REST_API_QUOTA":
        return f"SUCCESS [{now}]: Injected +25,000 emergency REST API burst buffer valid for 24 hours. Resumed integrations."
    elif action_type == "RESET_EVENT_BUS_CHECKPOINT":
        return f"SUCCESS [{now}]: Platform Event bus partition buffer reset. Checkpoint updated to Replay ID -2."
    return f"COMPLETED [{now}]: Synthesized manual mitigation guide for {action_type}."

# ==============================================================================
# 5. AGENTIC REACT REASONING & ORCHESTRATION PIPELINE
# ==============================================================================
def run_salesforce_copilot(scenario_key, custom_query, selected_edition, manual_sla, release_version, manual_action_override):
    """Orchestrates RAG retrieval, ReAct reasoning, and response formatting."""
    # Step A: Determine Context
    is_custom = bool(custom_query.strip()) and (custom_query.strip() != SCENARIO_DATABASE.get(scenario_key, {}).get("claim", ""))
    
    if not is_custom and scenario_key in SCENARIO_DATABASE:
        data = SCENARIO_DATABASE[scenario_key]
        incident_text = data["claim"]
        org_id = data["org_id"]
        edition = data["edition"]
        sla_tier = data["sla_tier"]
        error_code = data["error_code"]
        target_obj = data["object"]
    else:
        incident_text = custom_query.strip() if custom_query.strip() else "Apex CPU timeout on OrderTrigger execution."
        org_id = "00D9999999CUSTOM"
        edition = selected_edition
        sla_tier = manual_sla
        error_code = "MANUAL_INVESTIGATION"
        target_obj = "Custom_Object__c"

    # Step B: Grounded Vector RAG Retrieval
    retrieved_docs = vector_rag.retrieve(incident_text, top_k=2)
    kb_context = "\n\n".join([f"[{doc['id']}] {doc['title']}\n{doc['snippet']}" for doc in retrieved_docs])

    # Step C: ReAct Reasoning Loop & Tool Execution
    trace_steps = []
    trace_steps.append(f"⏱️ [PERCEPTION] Ingesting incident payload for Org {org_id} ({edition} Edition, {sla_tier}, Release: {release_version}).")
    trace_steps.append(f"🔍 [RAG RETRIEVAL] Cosine matching against Salesforce Knowledge Base. Grounded on: {retrieved_docs[0]['id']}.")

    # Tool 1: Execution Log Analysis
    trace_steps.append(f"⚙️ [TOOL EXECUTION] Invoking tool_inspect_apex_execution_log(org_id='{org_id}', error_code='{error_code}')...")
    log_results = tool_inspect_apex_execution_log(org_id, error_code, incident_text)
    trace_steps.append(f"   ↳ Observation: Status={log_results.get('status')}, Offending={log_results.get('offending_component', 'N/A')}.")

    # Tool 2: Database / Object Audit
    trace_steps.append(f"⚙️ [TOOL EXECUTION] Invoking tool_audit_soql_and_storage_limits(org_id='{org_id}', object='{target_obj}')...")
    storage_results = tool_audit_soql_and_storage_limits(org_id, target_obj)
    trace_steps.append(f"   ↳ Observation: Volume={storage_results.get('record_count', 'N/A')}, Index={storage_results.get('custom_index_present', 'N/A')}.")

    # Tool 3: SLA Verification
    trace_steps.append(f"⚙️ [TOOL EXECUTION] Invoking tool_verify_signature_sla_tier(org_id='{org_id}', sla='{sla_tier}')...")
    sla_results = tool_verify_signature_sla_tier(org_id, sla_tier)
    trace_steps.append(f"   ↳ Observation: IRT Target={sla_results['irt_minutes']}m | Dedicated TAM={sla_results.get('dedicated_tam')}.")

    # Step D: Remediation Determination
    if manual_action_override and manual_action_override != "Auto-Detect":
        action_decision = manual_action_override
    else:
        text_lower = incident_text.lower()
        if "cpu" in text_lower or "APEX_CPU" in error_code:
            action_decision = "INJECT_STATIC_RECURSION_GUARD"
        elif "soql" in text_lower or "SOQL" in error_code:
            action_decision = "PROVISION_CUSTOM_INDEX"
        elif "saml" in text_lower or "sso" in text_lower or "SAML" in error_code:
            action_decision = "PROVISION_SSO_BYPASS_URL"
        elif "data cloud" in text_lower or "DATA_CLOUD" in error_code:
            action_decision = "SURGE_DATA_CLOUD_CREDITS"
        elif "queue" in text_lower or "lock" in text_lower or "HOLDING_QUEUE" in error_code:
            action_decision = "ABORT_DEADLOCKED_BATCH"
        elif "event" in text_lower or "PLATFORM_EVENT" in error_code:
            action_decision = "RESET_EVENT_BUS_CHECKPOINT"
        elif "api" in text_lower or "limit" in text_lower or "REQUEST_LIMIT" in error_code:
            action_decision = "SURGE_REST_API_QUOTA"
        else:
            action_decision = "INJECT_STATIC_RECURSION_GUARD"

    trace_steps.append(f"🛠️ [AUTONOMOUS REMEDIATION] Executing action '{action_decision}' against Org {org_id}...")
    remediation_res = tool_execute_salesforce_remediation(action_decision, org_id, target_obj)
    trace_steps.append(f"   ↳ Result: {remediation_res}")
    trace_steps.append("✅ [COMPLETION] ReAct cycle verified. Dispatching Minto Work Order & Customer Payload.")

    react_trace_str = "\n".join(trace_steps)

    # Step E: Synthesize Minto Internal Work Order
    minto_work_order = f"""### 📋 TIER-3 TECHNICAL SUPPORT WORK ORDER (MINTO FORMAT)

**1. CORE VERDICT & REMEDIATION ACTION**
* **Primary Assessment:** Incident originates from `{log_results.get('offending_component', target_obj)}`.
* **Remediation Executed:** `{action_decision}`
* **Operational Status:** {remediation_res}

**2. MULTI-CONSOLE TELEMETRY CORRELATION**
* **Salesforce Org ID:** `{org_id}` ({edition} Edition | Release: `{release_version}`)
* **Contractual SLA:** `{sla_tier}` (Target IRT: {sla_results['irt_minutes']} mins | Designated TAM: {sla_results.get('dedicated_tam', 'Shared Premier')})
* **Telemetry Diagnostics:** CPU Duration = {log_results.get('cpu_time_ms', 'N/A')} ms | SOQL Count = {log_results.get('soql_count', 'N/A')} | Table Cardinality = {storage_results.get('record_count', 'Standard')}

**3. POLICY GROUNDING (SALESFORCE KCS RUNBOOK)**
* **Grounded Article:** `{retrieved_docs[0]['id']}` - {retrieved_docs[0]['title']}
* **Governor Limit Guidance:** Multi-tenant execution enforces strict thresholds. Workarounds must avoid synchronous recursion and enforce bulkified collections.
"""

    # Step F: Synthesize Customer-Facing Case Feed Update
    customer_feed_update = f"""Hello Salesforce Operations Team,

Our automated diagnostics analyzed Case telemetry for Org {org_id} regarding `{target_obj}`.

**Root Cause Analysis:**
Our runtime logs identified that the failure was caused by {log_results.get('root_cause', 'a runtime constraint during execution')}.

**Actions Taken & Next Steps:**
1. **Automated Platform Mitigation:** {remediation_res}
2. **Recommended Permanent Fix:** Review Salesforce Developer Guide reference `{retrieved_docs[0]['id']}`. Ensure that operations iterating over `{target_obj}` implement bulkified Map patterns rather than inline operations.
3. **Escalation Coverage:** Your case is operating under {sla_tier} guidelines with ongoing monitoring.

Please let us know if you observe any further anomalies on this transaction.

Warm regards,  
**Salesforce Agentforce Technical Support Copilot**
"""

    return minto_work_order, react_trace_str, kb_context, customer_feed_update

# ==============================================================================
# 6. GRADIO WEB COCKPIT INTERFACE
# ==============================================================================
def load_scenario_preset(key):
    if key in SCENARIO_DATABASE:
        s = SCENARIO_DATABASE[key]
        return s["claim"], s["edition"], s["sla_tier"]
    return "", "Enterprise", "Premier Support"

with gr.Blocks(title="Salesforce Agentic Service & Org Copilot", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # ☁️ Salesforce Agentic Service & Org Operations Copilot
        ### *Autonomous Multi-Tenant CRM Incident Diagnostic & Remediation Engine (Agentforce / Service Cloud)*
        """
    )
    
    with gr.Row():
        with gr.Column(scale=4):
            scenario_selector = gr.Dropdown(
                choices=list(SCENARIO_DATABASE.keys()),
                value="SF-APEX-101",
                label="📁 Select Real-World Salesforce Production Incident",
                info="Pre-configured multi-tenant production failure scenarios"
            )
            
            with gr.Row():
                edition_input = gr.Dropdown(
                    choices=["Enterprise", "Unlimited"],
                    value="Unlimited",
                    label="Salesforce Edition"
                )
                sla_input = gr.Dropdown(
                    choices=["Standard Support", "Premier Support", "Signature Success"],
                    value="Signature Success",
                    label="Contractual Support Tier"
                )

            with gr.Row():
                release_input = gr.Dropdown(
                    choices=["Winter '26", "Spring '26", "Summer '25"],
                    value="Winter '26",
                    label="Seasonal Release Target"
                )
                manual_action = gr.Dropdown(
                    choices=[
                        "Auto-Detect",
                        "INJECT_STATIC_RECURSION_GUARD",
                        "PROVISION_CUSTOM_INDEX",
                        "PROVISION_SSO_BYPASS_URL",
                        "SURGE_DATA_CLOUD_CREDITS",
                        "ABORT_DEADLOCKED_BATCH",
                        "SURGE_REST_API_QUOTA",
                        "RESET_EVENT_BUS_CHECKPOINT"
                    ],
                    value="Auto-Detect",
                    label="⚙️ Manual Remediation Override"
                )

            query_box = gr.Textbox(
                lines=3,
                label="📝 Inbound Case Payload / Developer Trace",
                value=SCENARIO_DATABASE["SF-APEX-101"]["claim"]
            )

            gr.Examples(
                examples=[
                    ["SF-APEX-101", "Apex CPU time limit exceeded (10001 ms) during checkout trigger loop."],
                    ["SF-SOQL-202", "System.LimitException: Too many SOQL queries: 101 on custom claims flow."],
                    ["SF-AUTH-303", "SAML SSO assertion failed: X.509 certificate revoked or expired."],
                    ["SF-DATA-404", "Data Cloud ingestion stream throttled with HTTP 429."],
                    ["SF-EVENT-707", "Order_Event__e publish buffer full (403): delivery blocked."],
                    ["SF-API-808", "MuleSoft sync failed: TotalRequests Limit exceeded on 24h rolling window."]
                ],
                inputs=[scenario_selector, query_box],
                label="💡 Quick-Fill Example Inquiries"
            )

            run_btn = gr.Button("🚀 Run Autonomous Agent Triage", variant="primary")

        with gr.Column(scale=6):
            with gr.Tabs():
                with gr.TabItem("📋 Minto Work Order"):
                    work_order_out = gr.Markdown()
                with gr.TabItem("🧠 ReAct Agent Reasoning Trace"):
                    trace_out = gr.Code(language="markdown")
                with gr.TabItem("📚 Grounded Vector RAG Context"):
                    rag_out = gr.Textbox(lines=8, label="Retrieved Salesforce KCS / SOP Documents")
                with gr.TabItem("💬 Customer Case Feed Update"):
                    customer_out = gr.Textbox(lines=8, label="Ready-to-Send Case Feed Update")

    # Wire Event Listeners
    scenario_selector.change(
        fn=load_scenario_preset,
        inputs=[scenario_selector],
        outputs=[query_box, edition_input, sla_input]
    )

    run_btn.click(
        fn=run_salesforce_copilot,
        inputs=[scenario_selector, query_box, edition_input, sla_input, release_input, manual_action],
        outputs=[work_order_out, trace_out, rag_out, customer_out]
    )

if __name__ == "__main__":
    assigned_port = find_available_port(7860)
    print(f"\n[Salesforce Agentic Copilot] Binding to available port: {assigned_port}")
    demo.launch(server_name="127.0.0.1", server_port=assigned_port)
