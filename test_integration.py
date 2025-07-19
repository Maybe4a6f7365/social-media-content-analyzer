#!/usr/bin/env python3

import asyncio
import logging
import sys
from typing import Dict, List

import httpx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleTester:

    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.base_url = base_url
        self.session = httpx.AsyncClient(timeout=60.0)

    async def test_basic_functionality(self) -> List[bool]:
        scenarios = [
            {
                "id": "test_001",
                "text": "The weather is nice today.",
                "language": "en"
            },
            {
                "id": "test_002", 
                "text": "Das Wetter ist heute schön.",
                "language": "de"
            },
            {
                "id": "test_003",
                "text": "I think we should go for a walk.",
                "language": "en"
            },
            {
                "id": "test_004",
                "text": "What time is the meeting?",
                "language": "en"
            },
            {
                "id": "test_005",
                "text": "Wir müssen unsere Männlichkeit wieder entdecken. Denn nur, wenn wir unsere Männlichkeit wieder entdecken, werden wir mannhaft. Und nur, wenn wir mannhaft werden, werden wir wehrhaft, und wir müssen wehrhaft werden, liebe Freunde.",
                "language": "de"
            }
        ]
        
        results = []
        for scenario in scenarios:
            result = await self._test_scenario(scenario)
            results.append(result)
        return results

    async def test_mixed_content(self) -> List[bool]:
        scenarios = [
            {
                "id": "test_006",
                "text": "NASA announced today that the Perseverance rover has discovered organic molecules on Mars.",
                "language": "en"
            },
            {
                "id": "test_007",
                "text": "I believe renewable energy is the future of our planet.",
                "language": "en"
            },
            {
                "id": "test_008",
                "text": "Why do we still use fossil fuels when solar power is available?",
                "language": "en"
            },
            {
                "id": "test_009",
                "text": "Die Deutsche Bahn hat heute Verspätungen gemeldet.",
                "language": "de"
            },
            {
                "id": "test_010",
                "text": "Ich finde, dass öffentlicher Verkehr ausgebaut werden sollte.",
                "language": "de"
            },
            {
                "id": "test_011",
                "text": "The stock market reached a new high today.",
                "language": "en"
            },
            {
                "id": "test_012",
                "text": "In my opinion, social media has changed how we communicate.",
                "language": "en"
            },
            {
                "id": "test_013",
                "text": "How can we reduce plastic waste in our daily lives?",
                "language": "en"
            },
            {
                "id": "test_014",
                "text": "Der Klimawandel ist eine der größten Herausforderungen unserer Zeit.",
                "language": "de"
            },
            {
                "id": "test_015",
                "text": "Apple released the new iPhone with improved camera features.",
                "language": "en"
            },
            {
                "id": "test_016",
                "text": "I think artificial intelligence will transform healthcare.",
                "language": "en"
            },
            {
                "id": "test_017",
                "text": "Should governments invest more in education?",
                "language": "en"
            },
            {
                "id": "test_018",
                "text": "Die Digitalisierung bietet viele neue Möglichkeiten.",
                "language": "de"
            },
            {
                "id": "test_019",
                "text": "The Berlin Wall fell on November 9, 1989.",
                "language": "en"
            },
            {
                "id": "test_020",
                "text": "I believe universal basic income could solve many problems.",
                "language": "en"
            },
            {
                "id": "test_021",
                "text": "Warum ist die Miete in Großstädten so hoch?",
                "language": "de"
            },
            {
                "id": "test_022",
                "text": "Scientists discovered a new species of deep-sea creatures.",
                "language": "en"
            },
            {
                "id": "test_023",
                "text": "In my view, traditional media is still more reliable.",
                "language": "en"
            },
            {
                "id": "test_024",
                "text": "How can we improve public transportation?",
                "language": "en"
            },
            {
                "id": "test_025",
                "text": "Die Energiewende ist wichtig für unsere Zukunft.",
                "language": "de"
            }
        ]
        
        results = []
        for scenario in scenarios:
            result = await self._test_scenario(scenario)
            results.append(result)
        return results

    async def test_edge_cases(self) -> List[bool]:
        scenarios = [
            {
                "id": "test_026",
                "text": "Hi",
                "language": "en"
            },
            {
                "id": "test_027",
                "text": "",
                "language": "en"
            },
            {
                "id": "test_028",
                "text": "This is a very long text that contains many sentences. " * 50,
                "language": "en"
            },
            {
                "id": "test_029",
                "text": "Text with special chars: @#$%^&*()_+-=[]{}|;':\",./<>?",
                "language": "en"
            },
            {
                "id": "test_030",
                "text": "1234567890",
                "language": "en"
            },
            {
                "id": "test_031",
                "text": "🚀🌟💡🎯🔥",
                "language": "en"
            },
            {
                "id": "test_032",
                "text": "Hello! Wie geht es dir? Bonjour!",
                "language": "en"
            },
            {
                "id": "test_033",
                "text": "This is a test. This is a test. This is a test.",
                "language": "en"
            }
        ]
        
        results = []
        for scenario in scenarios:
            result = await self._test_scenario(scenario)
            results.append(result)
        return results

    async def test_error_conditions(self) -> List[bool]:
        scenarios = [
            {
                "id": "test_034",
                "data": {"post_text": "Test content", "language": "en"}
            },
            {
                "id": "test_035",
                "data": {"post_id": "test", "language": "en"}
            },
            {
                "id": "test_036",
                "data": {"post_id": "test", "post_text": "Test", "language": "fr"}
            },
            {
                "id": "test_037",
                "data": {"post_id": "a" * 1000, "post_text": "Test", "language": "en"}
            },
            {
                "id": "test_038",
                "data": {"post_id": "test", "post_text": "Test", "language": "en", "extra": None}
            }
        ]
        
        results = []
        for scenario in scenarios:
            result = await self._test_error_scenario(scenario)
            results.append(result)
        return results

    async def test_performance(self) -> List[bool]:
        scenarios = [
            {
                "id": "perf_concurrent",
                "count": 5,
                "description": "5 concurrent requests"
            },
            {
                "id": "perf_sequential",
                "count": 10,
                "description": "10 sequential requests"
            },
            {
                "id": "perf_mixed_lang",
                "count": 3,
                "description": "Mixed language requests"
            }
        ]
        
        results = []
        for scenario in scenarios:
            result = await self._test_performance_scenario(scenario)
            results.append(result)
        return results

    async def _test_scenario(self, scenario: Dict) -> bool:
        try:
            logger.info(f"Testing: {scenario['id']}")
            
            test_data = {
                "post_id": scenario["id"],
                "post_text": scenario["text"],
                "language": scenario["language"]
            }
            
            response = await self.session.post(
                f"{self.base_url}/evaluate",
                json=test_data,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                post_type = result.get("post_analysis", {}).get("post_type", "")
                is_spam = result.get("post_analysis", {}).get("is_spam", False)
                
                logger.info(f"  Result: {post_type}, spam: {is_spam}")
                return True
            else:
                logger.error(f"  Failed with status: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"  Error: {e}")
            return False

    async def _test_error_scenario(self, scenario: Dict) -> bool:
        try:
            logger.info(f"Testing error: {scenario['id']}")
            
            response = await self.session.post(
                f"{self.base_url}/evaluate",
                json=scenario["data"],
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code in [422, 400]:
                logger.info(f"  Handled correctly with status: {response.status_code}")
                return True
            else:
                logger.error(f"  Unexpected status: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"  Error: {e}")
            return False

    async def _test_performance_scenario(self, scenario: Dict) -> bool:
        try:
            logger.info(f"Testing performance: {scenario['id']}")
            
            test_data = {
                "post_id": "perf_test",
                "post_text": "This is a performance test message.",
                "language": "en"
            }
            
            if scenario["id"] == "perf_concurrent":
                tasks = []
                for i in range(scenario["count"]):
                    task = self.session.post(
                        f"{self.base_url}/evaluate",
                        json={**test_data, "post_id": f"perf_test_{i}"},
                        headers={"Content-Type": "application/json"}
                    )
                    tasks.append(task)
                
                responses = await asyncio.gather(*tasks, return_exceptions=True)
                success_count = sum(1 for r in responses if isinstance(r, httpx.Response) and r.status_code == 200)
                
            elif scenario["id"] == "perf_sequential":
                success_count = 0
                for i in range(scenario["count"]):
                    response = await self.session.post(
                        f"{self.base_url}/evaluate",
                        json={**test_data, "post_id": f"seq_test_{i}"},
                        headers={"Content-Type": "application/json"}
                    )
                    if response.status_code == 200:
                        success_count += 1
                        
            else:
                languages = ["en", "de", "en"]
                success_count = 0
                for i, lang in enumerate(languages):
                    response = await self.session.post(
                        f"{self.base_url}/evaluate",
                        json={**test_data, "post_id": f"lang_test_{i}", "language": lang},
                        headers={"Content-Type": "application/json"}
                    )
                    if response.status_code == 200:
                        success_count += 1
            
            success_rate = success_count / scenario["count"]
            logger.info(f"  Success rate: {success_rate:.2%} ({success_count}/{scenario['count']})")
            return success_rate >= 0.8
            
        except Exception as e:
            logger.error(f"  Error: {e}")
            return False

    async def run_all_tests(self) -> Dict[str, List[bool]]:
        logger.info("Starting comprehensive tests...")
        
        test_results = {
            "Basic Functionality": await self.test_basic_functionality(),
            "Mixed Content": await self.test_mixed_content(),
            "Edge Cases": await self.test_edge_cases(),
            "Error Conditions": await self.test_error_conditions(),
            "Performance": await self.test_performance()
        }
        
        return test_results

    async def close(self):
        await self.session.aclose()


async def main():
    base_url = "http://127.0.0.1:8000"
    if len(sys.argv) > 1:
        base_url = sys.argv[1]
        logger.info(f"Using custom base URL: {base_url}")
    
    tester = SimpleTester(base_url)
    
    try:
        results = await tester.run_all_tests()
        
        logger.info("="*80)
        logger.info("TEST RESULTS")
        logger.info("="*80)
        
        total_tests = 0
        total_passed = 0
        
        for category, test_results in results.items():
            passed = sum(test_results)
            total = len(test_results)
            total_tests += total
            total_passed += passed
            
            success_rate = passed / total if total > 0 else 0
            status = "PASS" if success_rate >= 0.8 else "FAIL"
            
            logger.info(f"{status} {category}: {passed}/{total} ({success_rate:.1%})")
        
        overall_success_rate = total_passed / total_tests if total_tests > 0 else 0
        logger.info(f"Overall: {total_passed}/{total_tests} tests passed ({overall_success_rate:.1%})")
        
        if overall_success_rate >= 0.8:
            logger.info("Testing completed successfully!")
        else:
            logger.error("Many tests failed. Review the logs above.")
            
    finally:
        await tester.close()


if __name__ == "__main__":
    asyncio.run(main()) 