import Link from "next/link";
import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Brain Treebank Leaderboard",
  description: "A leaderboard for brain decoder AIs",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="bg-gray-900 text-white">
        {/* Navbar */}
        <nav className="bg-gray-800 p-4">
          <div className="container mx-auto flex justify-between items-center">
            <h1 className="text-xl font-bold">Brain Treebank Leaderboard</h1>
            
            {/* Navigation Links */}
            <div className="flex items-center space-x-6">
              <Link href="/" className="text-gray-300 hover:text-white">Home</Link>
              <Link href="/leaderboards" className="text-gray-300 hover:text-white">Leaderboards</Link>
              <Link href="/about" className="text-gray-300 hover:text-white">About</Link>
              
              {/* Log In Button */}
              <Link href="/login">
                <button className="ml-6 bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg">
                  Log In
                </button>
              </Link>
            </div>
          </div>
        </nav>

        {/* Page Content */}
        <main className="p-6">{children}</main>
      </body>
    </html>
  );
}
