"""
Kafka Producer Service for publishing events
"""
import logging
import json
from typing import Optional
from aiokafka import AIOKafkaProducer
from aiokafka.errors import KafkaError

from app.config import settings
from app.models.schemas import UserInteractionEvent

logger = logging.getLogger(__name__)


class KafkaProducerService:
    """
    Asynchronous Kafka producer for publishing user interaction events
    """
    
    def __init__(self):
        self.producer: Optional[AIOKafkaProducer] = None
        self.is_running = False
    
    async def start(self):
        """Initialize and start the Kafka producer"""
        try:
            self.producer = AIOKafkaProducer(
                bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                compression_type='gzip',
                acks='all',  # Wait for all replicas
                retries=3,
                max_in_flight_requests_per_connection=5
            )
            
            await self.producer.start()
            self.is_running = True
            logger.info(f"Kafka producer started: {settings.KAFKA_BOOTSTRAP_SERVERS}")
            
        except KafkaError as e:
            logger.error(f"Failed to start Kafka producer: {str(e)}")
            logger.warning("Service will continue without Kafka connectivity")
            self.is_running = False
        except Exception as e:
            logger.error(f"Unexpected error starting Kafka producer: {str(e)}")
            self.is_running = False
    
    async def stop(self):
        """Stop the Kafka producer"""
        if self.producer:
            try:
                await self.producer.stop()
                self.is_running = False
                logger.info("Kafka producer stopped")
            except Exception as e:
                logger.error(f"Error stopping Kafka producer: {str(e)}")
    
    async def publish_interaction(self, interaction: UserInteractionEvent):
        """
        Publish user interaction event to Kafka
        """
        if not self.is_running or not self.producer:
            logger.warning("Kafka producer not available, skipping event publish")
            return
        
        try:
            # Convert Pydantic model to dict
            event_data = {
                "user_id": interaction.user_id,
                "product_id": interaction.product_id,
                "interaction_type": interaction.interaction_type,
                "timestamp": interaction.timestamp.isoformat(),
                "metadata": interaction.metadata
            }
            
            # Publish to Kafka
            await self.producer.send_and_wait(
                settings.KAFKA_INTERACTION_TOPIC,
                value=event_data,
                key=interaction.user_id.encode('utf-8')
            )
            
            logger.debug(
                f"Published event to Kafka: {interaction.user_id} - {interaction.product_id}"
            )
            
        except KafkaError as e:
            logger.error(f"Kafka error publishing event: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error publishing event: {str(e)}")
            raise
    
    def is_connected(self) -> bool:
        """Check if producer is connected and running"""
        return self.is_running and self.producer is not None
