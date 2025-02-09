"use client";

import { useState, useEffect } from "react";
import Link from "next/link";

export default function Leaderboards() {
  const [tasks, setTasks] = useState<{ id: string; name: string; question: string }[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch("/leaderboard_data.json") // ✅ Load tasks dynamically
      .then((res) => res.json())
      .then((data) => {
        if (data.tasks && Array.isArray(data.tasks)) {
          const formattedTasks = data.tasks.map((taskId: string) => ({
            id: taskId,
            name: taskId.replace("-", " ").replace(/\b\w/g, (c) => c.toUpperCase()), // Capitalize
            question: data.descriptions?.[taskId] || "Can a decoder predict patterns in the brain's activity?"
          }));
          setTasks(formattedTasks);
        }
        setLoading(false);
      })
      .catch((err) => console.error("Failed to load tasks:", err));
  }, []);

  if (loading) return <p className="text-white text-center">Loading...</p>;

  return (
    <div className="min-h-screen bg-gray-900 text-white flex flex-col items-center p-6">
      <h1 className="text-4xl font-bold mb-2">Neural Decoding Leaderboards</h1>
      <p className="text-gray-400 max-w-3xl text-center mb-4">
        Each leaderboard corresponds to a unique neuroscience decoding task using the Brain Treebank dataset. 
        Click a leaderboard to see model rankings, or submit your model for evaluation.
      </p>

      <div className="w-full max-w-3xl space-y-6">
        {tasks.length > 0 ? (
          tasks.map((task) => (
            <div key={task.id} className="bg-gray-800 p-6 rounded-lg shadow-md flex flex-col md:flex-row md:items-center justify-between">
              {/* Task Details */}
              <div className="mb-6 md:mb-0 flex-1">
                <h2 className="text-xl font-semibold">{task.name}</h2>
                <p className="text-gray-400 text-sm">{task.question}</p>
              </div>

              {/* Buttons: View Leaderboard & Submit Model */}
              <div className="flex flex-wrap md:flex-nowrap gap-2 md:gap-4">
                <Link href={`/leaderboards/${task.id}`} className="w-40 text-center bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded-lg">
                  View & Submit
                </Link>
              </div>
            </div>
          ))
        ) : (
          <p className="text-gray-400 text-center">No tasks available.</p>
        )}
      </div>
    </div>
  );
}
