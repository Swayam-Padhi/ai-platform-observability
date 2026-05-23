from dotenv import load_dotenv
import os
import base64

from crewai import Agent, Task, Crew
from langfuse import get_client
from openinference.instrumentation.crewai import CrewAIInstrumentor
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry import trace

# Load env variables
load_dotenv()

# Build Basic Auth header from Langfuse keys
auth = base64.b64encode(
    f"{os.environ['LANGFUSE_PUBLIC_KEY']}:{os.environ['LANGFUSE_SECRET_KEY']}".encode()
).decode()

trace.set_tracer_provider(TracerProvider())

otlp_exporter = OTLPSpanExporter(
    endpoint="http://localhost:3000/api/public/otel/v1/traces",
    headers={"Authorization": f"Basic {auth}"},
    timeout=60
)

span_processor = BatchSpanProcessor(
    otlp_exporter,
    max_export_batch_size=256,     # Smaller batches export faster
    export_timeout_millis=60000,   # Give more time per batch
    schedule_delay_millis=3000     # Send more frequently
)

trace.get_tracer_provider().add_span_processor(span_processor)

CrewAIInstrumentor().instrument()

langfuse = get_client()


# Ollama local model
llm = "ollama/phi3"

# Triage Agent
triage_agent = Agent(
    role="Incident Triage Engineer",
    goal="Quickly analyze system issues and identify root cause signals",
    backstory=(
        "You are an SRE expert who handles production outages. "
        "You focus on logs, metrics, and system behavior to identify root causes."
    ),
    llm=llm,
    verbose=True
)

# RCA Agent
rca_agent = Agent(
    role="Root Cause Analyst",
    goal="Determine the most likely root cause of system failures",
    backstory=(
        "You specialize in debugging distributed systems, Kubernetes, and cloud-native apps. "
        "You connect symptoms to actual infrastructure problems."
    ),
    llm=llm,
    verbose=True
)

# Incident Report Agent
report_agent = Agent(
    role="Incident Report Writer",
    goal="Write clear postmortem reports for engineering teams",
    backstory=(
        "You turn technical findings into structured postmortem reports "
        "for engineering and leadership teams."
    ),
    llm=llm,
    verbose=True
)

# Triage Task
triage_task = Task(
    description=(
        "A Kubernetes cluster is experiencing high pod restarts, "
        "increased latency, and CPU spikes. "
        "Analyze symptoms and list possible causes."
    ),
    expected_output="List of potential system issues and signals",
    agent=triage_agent
)

# RCA Task
rca_task = Task(
    description=(
        "Based on the triage findings, determine the most likely root cause "
        "of the Kubernetes instability."
    ),
    expected_output="Single most likely root cause with explanation",
    agent=rca_agent
)

# Incident Task
report_task = Task(
    description=(
        "Write a production incident report based on the root cause analysis. "
        "Include: summary, impact, root cause, and prevention steps."
    ),
    expected_output="Structured incident postmortem report",
    agent=report_agent
)

# Crew Definition
crew = Crew(
    agents=[triage_agent, rca_agent, report_agent],
    tasks=[triage_task, rca_task, report_task],
    verbose=True
)

# Execute Crew
result = crew.kickoff()

print("\n====================")
print("FINAL CREW OUTPUT")
print("====================\n")

print(result)