"""
Kafka Consumer Service for processing events
"""
import logging
import json
import asyncio
from typing import Optional
from datetime import datetime
from aiokafka import AIOKafkaConsumer
from aiokafka.errors import KafkaError

from app.config import settings
from app.models.schemas import UserInteractionEvent, InteractionType

logger = logging.getLogger(__name__)


class KafkaConsumerService:
    """
    Asynchronous Kafka consumer for processing user interaction events
    """
    
    def __init__(self, recommendation_engine):
        self.consumer: Optional[AIOKafkaConsumer] = None
        self.recommendation_engine = recommendation_engine
        self.is_running = False
        self.consumer_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Initialize and start the Kafka consumer"""
        try:
            self.consumer = AIOKafkaConsumer(
                settings.KAFKA_INTERACTION_TOPIC,
                bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS,
                group_id=settings.KAFKA_CONSUMER_GROUP,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                auto_offset_reset=settings.KAFKA_AUTO_OFFSET_RESET,
                enable_auto_commit=True,
                auto_commit_interval_ms=1000
            )
            
            await self.consumer.start()
            self.is_running = True
            
            # Start consuming in background
            self.consumer_task = asyncio.create_task(self._consume_messages())
            
            logger.info(
                f"Kafka consumer started: topic={settings.KAFKA_INTERACTION_TOPIC}, "
                f"group={settings.KAFKA_CONSUMER_GROUP}"
            )
            
        except KafkaError as e:
            logger.error(f"Failed to start Kafka consumer: {str(e)}")
            logger.warning("Service will continue without Kafka event processing")
            self.is_running = False
        except Exception as e:
            logger.error(f"Unexpected error starting Kafka consumer: {str(e)}")
            self.is_running = False
    
    async def stop(self):
        """Stop the Kafka consumer"""
        self.is_running = False
        
        if self.consumer_task:
            self.consumer_task.cancel()
            try:
                await self.consumer_task
            except asyncio.CancelledError:
                pass
        
        if self.consumer:
            try:
                await self.consumer.stop()
                logger.info("Kafka consumer stopped")
            except Exception as e:
                logger.error(f"Error stopping Kafka consumer: {str(e)}")
    
    async def _consume_messages(self):
        """
        Continuously consume and process messages from Kafka
        Runs in background task
        """
        logger.info("Starting to consume Kafka messages...")
        
        try:
            async for msg in self.consumer:
                if not self.is_running:
                    break
                
                try:
                    await self._process_message(msg.value)
                except Exception as e:
                    logger.error(f"Error processing message: {str(e)}")
                    # Continue processing next messages
                    
        except asyncio.CancelledError:
            logger.info("Kafka consumer task cancelled")
        except Exception as e:
            logger.error(f"Error in consumer loop: {str(e)}")
    
    async def _process_message(self, message_data: dict):
        """
        Process a single Kafka message
        Converts to UserInteractionEvent and updates recommendation engine
        """
        try:
            # Parse timestamp
            timestamp = datetime.fromisoformat(message_data.get('timestamp'))
            
            # Create UserInteractionEvent
            interaction = UserInteractionEvent(
                user_id=message_data['user_id'],
                product_id=message_data['product_id'],
                interaction_type=InteractionType(message_data['interaction_type']),
                timestamp=timestamp,
                metadata=message_data.get('metadata', {})
            )
            
            # Process through recommendation engine
            await self.recommendation_engine.process_interaction(interaction)
            
            logger.debug(
                f"Processed Kafka message: {interaction.user_id} - {interaction.product_id}"
            )
            
        except Exception as e:
            logger.error(f"Error processing Kafka message: {str(e)}")
            raise
