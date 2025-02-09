"use client";

import { useState, useEffect } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";

export default function TaskLeaderboard() {
  const params = useParams();
  const [taskId, setTaskId] = useState<string | null>(null);
  const [leaderboardData, setLeaderboardData] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [validTasks, setValidTasks] = useState<string[]>([]);
  const [taskName, setTaskName] = useState<string>("");

  useEffect(() => {
    fetch("/leaderboard_data.json")
      .then((res) => res.json())
      .then((data) => {
        setValidTasks(data.tasks || []);

        if (params && typeof params.task === "string" && data.tasks.includes(params.task)) {
          setTaskId(params.task);
          setTaskName(params.task.replace("-", " ").replace(/\b\w/g, (c) => c.toUpperCase())); // Capitalize
          setLeaderboardData((data[params.task] || []).sort((a, b) => b.rocAuc - a.rocAuc));
        }

        setLoading(false);
      })
      .catch((err) => console.error("Failed to load leaderboard:", err));
  }, [params]);

  if (!taskId) return <p className="text-white text-center">Invalid task.</p>;
  if (loading) return <p className="text-white text-center">Loading leaderboard data...</p>;

  return (
    <div className="min-h-screen bg-gray-900 text-white flex flex-col items-center p-6">
      <h1 className="text-4xl font-bold mb-6 text-center">Leaderboard for {taskName}</h1>

      <div className="bg-gray-800 p-6 rounded-lg shadow-lg w-full max-w-4xl">
        <table className="w-full text-left border-collapse">
          <thead>
            <tr className="bg-gray-700">
              <th className="p-3">Rank</th>
              <th className="p-3">Model</th>
              <th className="p-3">ROC AUC</th>
              <th className="p-3">Accuracy (%)</th>
              <th className="p-3">Org</th>
              <th className="p-3">Date</th>
            </tr>
          </thead>
          <tbody>
            {leaderboardData.length > 0 ? (
              leaderboardData.map((entry, index) => {
                let rankStyle = "";
                if (index === 0) rankStyle = "bg-yellow-500 text-black font-bold"; // Gold
                else if (index === 1) rankStyle = "bg-gray-400 text-black font-bold"; // Silver
                else if (index === 2) rankStyle = "bg-orange-500 text-black font-bold"; // Bronze

                return (
                  <tr key={index} className="border-b border-gray-700">
                    <td className={`p-3 text-center rounded-lg ${rankStyle}`}>{index + 1}</td>
                    <td className="p-3">{entry.name}</td>
                    <td className="p-3">{entry.rocAuc.toFixed(2)}</td>
                    <td className="p-3">{entry.accuracy}%</td>
                    <td className="p-3">{entry.org}</td>
                    <td className="p-3">{entry.date}</td>
                  </tr>
                );
              })
            ) : (
              <tr>
                <td colSpan={6} className="p-3 text-center text-gray-400">No submissions yet.</td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      {/* Submit Model Button */}
      <div className="w-full max-w-4xl flex justify-end mt-4">
        <Link href={`/leaderboards/${taskId}/submit`}>
          <button className="bg-green-500 hover:bg-green-600 text-white px-6 py-3 rounded-lg text-lg">
            Submit Model
          </button>
        </Link>
      </div>
    </div>
  );
}
