"use client";

import { useState, useEffect } from "react";
import { useParams } from "next/navigation";

export default function SubmitModel() {
  const params = useParams(); // Gets the params dynamically
  const [taskId, setTaskId] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [message, setMessage] = useState("");

  // ✅ Wait for `params.task` before using it
  useEffect(() => {
    if (params && typeof params.task === "string") {
      setTaskId(params.task);
    }
  }, [params]);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files[0]) {
      setFile(event.target.files[0]);
    }
  };

  const handleSubmit = (event: React.FormEvent) => {
    event.preventDefault();
    if (!file) {
      setMessage("Please select a file to upload.");
      return;
    }
    setMessage(`Successfully uploaded ${file.name} for ${taskId?.replace("-", " ")}`);
  };

  if (!taskId) return <p className="text-white text-center">Loading...</p>;

  return (
    <div className="min-h-screen bg-gray-900 text-white flex flex-col items-center p-6">
      <h1 className="text-4xl font-bold mb-4">Submit Model for {taskId.replace("-", " ")}</h1>
      
      <p className="text-gray-300 max-w-2xl text-center mb-6">
        Please submit your trained model file for evaluation. The file should be in **.zip** format containing:
        - Your model file (`.h5`, `.pt`, `.pkl`, etc.)
        - A README explaining how to run it
      </p>

      <form onSubmit={handleSubmit} className="bg-gray-800 p-6 rounded-lg shadow-lg">
        <input
          type="file"
          onChange={handleFileChange}
          className="mb-4 p-2 bg-gray-700 text-white rounded"
        />
        <button
          type="submit"
          className="bg-yellow-500 hover:bg-yellow-600 text-white px-6 py-2 rounded-lg"
        >
          Upload Model
        </button>
      </form>

      {message && <p className="mt-4 text-green-400">{message}</p>}
    </div>
  );
}
