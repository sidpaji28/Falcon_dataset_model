export default function UserDashboard() {
  return (
    <div className="p-8">
      <h1 className="text-3xl font-bold mb-4">User Dashboard</h1>
      <p>Interact with the JSO Agent to analyze your CV and extract matching job listings using the Gemini API.</p>
      <div className="mt-4 p-4 border rounded">
        <h2 className="text-xl font-semibold">Start Search</h2>
        <button className="mt-2 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600">Analyze CV & Start Search</button>
      </div>
    </div>
  );
}
