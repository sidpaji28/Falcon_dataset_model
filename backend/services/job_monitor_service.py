import asyncio
import logging
from typing import Callable
from backend.db.supabase_client import supabase_client

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

    async def run_monitoring_cycle(self):
        """
        Runs a single cycle of the job monitor.
        Designed to be called by a serverless Cron job (e.g. Vercel Cron).
        """
        logger.info("Running single job monitoring cycle.")
        processed_users = 0
        try:
            # Iterate over a list copy to prevent RuntimeError if dictionary changes size during iteration
            for user_id, prefs in list(self.agent.user_preferences.items()):
                logger.info(f"Regenerating queries for user {user_id}")

                # 1. Regenerate queries
                role = prefs.get("role")
                location = prefs.get("location")
                skills = prefs.get("skills", [])

                if role and location:
                    result = await self.agent.start_search(user_id, role, location, skills)
                    processed_users += 1

                    # 2. Check for new jobs (simulated here)
                    # In a real system, you would call external APIs or scrape sites using generated URLs

                    # 3. Update Supabase
                    if supabase_client:
                        # Actually use the supabase client to insert the newly found queries
                        data = {
                            "user_id": user_id,
                            "boolean_queries": result.get("boolean_queries", []),
                            "platform_links": result.get("platform_links", []),
                            "timestamp": "now"
                        }
                        # Simulating real Supabase insertion (assuming 'job_queries' table exists)
                        try:
                            supabase_client.table('job_queries').insert(data).execute()
                            logger.info(f"Successfully updated Supabase for user {user_id}")
                        except Exception as db_err:
                            logger.error(f"Failed to update Supabase for user {user_id}: {db_err}")

                    # Fallback callback if Supabase isn't configured
                    elif self.db_update_callback:
                        if asyncio.iscoroutinefunction(self.db_update_callback):
                            await self.db_update_callback(user_id, result)
                        else:
                            self.db_update_callback(user_id, result)
                    logger.info(f"Updated data for user {user_id}")

        except Exception as e:
            logger.error(f"Error in monitor cycle: {e}")
            raise

        return processed_users
