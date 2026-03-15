import asyncio
import logging
from typing import Callable

logger = logging.getLogger(__name__)

class JobMonitorService:
    """
    Background job monitoring service.
    Runs every 12 hours to regenerate queries, check for new jobs,
    and simulate updating Supabase.
    """
    def __init__(self, agent, db_update_callback: Callable = None):
        self.agent = agent
        self.db_update_callback = db_update_callback
        self.interval = 12 * 60 * 60  # 12 hours in seconds
        self._is_running = False
        self._task = None

    async def _monitor_loop(self):
        logger.info("Starting JobMonitorService loop.")
        while self._is_running:
            try:
                logger.info("Running job monitoring cycle.")

                # Iterate over a list copy to prevent RuntimeError if dictionary changes size during iteration
                for user_id, prefs in list(self.agent.user_preferences.items()):
                    logger.info(f"Regenerating queries for user {user_id}")

                    # 1. Regenerate queries
                    role = prefs.get("role")
                    location = prefs.get("location")
                    skills = prefs.get("skills", [])

                    if role and location:
                        result = await self.agent.start_search(user_id, role, location, skills)

                        # 2. Check for new jobs (simulated here)
                        # In a real system, you would call external APIs or scrape sites using generated URLs

                        # 3. Update Supabase
                        if self.db_update_callback:
                            if asyncio.iscoroutinefunction(self.db_update_callback):
                                await self.db_update_callback(user_id, result)
                            else:
                                self.db_update_callback(user_id, result)
                        logger.info(f"Updated data for user {user_id}")

            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")

            # Wait for next cycle
            logger.info(f"Sleeping for {self.interval} seconds.")
            await asyncio.sleep(self.interval)

    def start(self):
        if not self._is_running:
            self._is_running = True
            self._task = asyncio.create_task(self._monitor_loop())

    def stop(self):
        self._is_running = False
        if self._task:
            self._task.cancel()
