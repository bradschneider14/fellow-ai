import os
from fellowai.workers.base_worker import BaseRMQWorker

class ProducerWorker(BaseRMQWorker):
    def __init__(self):
        super().__init__(listen_queue="dummy", publish_queue="initiate_queue")

def main():
    if not os.environ.get("OPENAI_API_KEY") and not os.environ.get("XAI_API_KEY"):
        print("Please set the OPENAI_API_KEY or XAI_API_KEY environment variable.")
        return

    print("--- FellowAI: Async RMQ Workflow Producer ---")
    sample_pdf_url = "https://arxiv.org/pdf/2506.06718"

    producer = ProducerWorker()
    producer.connect()
    
    payload = {
        "pdf_source": sample_pdf_url,
        "static_retry_count": 0,
        "runtime_retry_count": 0
    }
    
    print(f"Publishing new project initiation for PDF: {sample_pdf_url}")
    producer.publish_output_to_queue(payload)
    print("Payload successfully delivered to `initiate_queue`. Ensure Docker RMQ and workers are running to process it!")

if __name__ == "__main__":
    main()
